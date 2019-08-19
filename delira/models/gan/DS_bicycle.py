import logging
import tensorflow as tf
import typing
import numpy as np

from delira.models.abstract_network import AbstractTfNetwork

tf.keras.backend.set_image_data_format('channels_first')
logger = logging.getLogger(__name__)


from delira.models.gan.cgan_models import Discriminator, Generator_add_all


def slerp(p0, p1, t, omega=None):
    if not omega:
        omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1



class DSGenerativeAdversarialNetworkBaseTf(AbstractTfNetwork):
    """Implementation of Diversity -Sensitive Conditional GAN for multimodal image to image translation
        based on https://arxiv.org/abs/1901.09024


        References
        ----------
        https://github.com/maga33/DSGAN

        See Also
        --------
        :class:`AbstractTfNetwork`

    """

    def __init__(self, image_size: int, input_c_dim: int, output_c_dim: int, df_dim: int,
                 gf_dim: int, latent_dim: int, n_latent: int, coeff_reconstruct: int,
                 coeff_ds: int, f_size: int, is_batchnorm: bool, **kwargs):
        """

        Constructs graph containing model definition and forward pass behavior

        """
        # register params by passing them as kwargs to parent class __init__
        super().__init__(image_size=image_size, input_c_dim=input_c_dim, output_c_dim=output_c_dim, df_dim=df_dim,
                         gf_dim=gf_dim, latent_dim=latent_dim, n_latent=n_latent, coeff_reconstruct=coeff_reconstruct,
                         coeff_ds=coeff_ds, f_size=f_size, is_batchnorm=is_batchnorm, **kwargs)

        is_bicycle = True

        self.image_size = image_size

        self.coeff_reconstruct = coeff_reconstruct
        self.coeff_ds = coeff_ds
        self.latent_dim = latent_dim
        self.n_latent = n_latent

        real_data = tf.placeholder(shape=[None, input_c_dim + output_c_dim, image_size, image_size], dtype=tf.float32)
        self.image_a = real_data[:, :input_c_dim, :, :]
        self.image_b = real_data[:, input_c_dim:input_c_dim + output_c_dim, :, :]
        self.z = tf.placeholder(shape=[None, latent_dim], dtype=tf.float32)
        self.z1 = tf.placeholder(shape=[None, latent_dim], dtype=tf.float32)

        self.discr, self.gen = self._build_models(output_c_dim, df_dim, gf_dim, latent_dim, f_size, is_batchnorm, is_bicycle)

        # Fake values generated corresponding to two randomly sampled 'z'
        fake_b = self.gen(self.image_a, self.z)
        fake_b1 = self.gen(self.image_a, self.z1)


        # Discriminator used  in conditional setting
        discr_ip_ab = tf.concat([self.image_a, self.image_b], 1)
        D_real = self.discr(discr_ip_ab)  # For real image_b
        discr_ip_ab = tf.concat([self.image_a, fake_b], 1)
        D_fake = self.discr(discr_ip_ab)  # For fake b



        self.inputs = [real_data, self.z, self.z1]
        self.outputs_train = [fake_b, fake_b1, D_real, D_fake]
        self.outputs_eval = [fake_b, fake_b1, D_real, D_fake]

        for key, value in kwargs.items():
            setattr(self, key, value)


    def _add_losses(self, losses: dict):
        """
        Adds losses to model that are to be used by optimizers or during evaluation

        Parameters
        ----------
        losses : dict
            dictionary containing all losses. Individual losses are averaged for discr_real, discr_fake and gen
        """
        if self._losses is not None and len(losses) != 0:
            logging.warning('Change of losses is not yet supported')
            raise NotImplementedError()
        elif self._losses is not None and len(losses) == 0:
            pass
        else:
            self._losses = {}

            total_loss_discr_fake = []
            total_loss_discr_real = []
            loss_gen = []
            eps = 0.00001

            for name, _loss in losses.items():
                if name == 'L2':
                    loss_val = tf.reduce_mean(_loss(tf.ones_like(self.outputs_train[2]), self.outputs_train[2]))
                    self._losses[name + 'discr_real'] = loss_val
                    total_loss_discr_real.append(loss_val)

                    loss_val = tf.reduce_mean(_loss(tf.zeros_like(self.outputs_train[3]), self.outputs_train[3]))
                    self._losses[name + 'discr_fake'] = loss_val
                    total_loss_discr_fake.append(loss_val)

                    loss_gen = tf.reduce_mean(_loss(tf.ones_like(self.outputs_train[3]), self.outputs_train[3]))
                    self._losses[name + 'gen'] = loss_gen


            for name, _loss in losses.items():
                if name == 'L1':
                    loss_L1 = tf.reduce_mean(_loss(self.image_b, self.outputs_train[0]))
                    self._losses[name + 'L1'] = loss_L1

                    loss_ds_img = tf.reduce_mean(tf.abs(self.outputs_train[0] - self.outputs_train[1]), axis=[1, 2, 3])

                    loss_ds_z = tf.reduce_mean(tf.abs(self.z - self.z1), axis=1)

                    loss_ds = - tf.reduce_mean(tf.divide(loss_ds_img, loss_ds_z + eps))
                    self._losses[name + 'ds_loss'] = loss_ds



            total_loss_discr = tf.reduce_mean([*total_loss_discr_real, *total_loss_discr_fake], axis=0) * 0.5
            self._losses['total_discr'] = total_loss_discr

            total_loss_gen = loss_gen + self.coeff_reconstruct * loss_L1 + self.coeff_ds * loss_ds
            self._losses['total_gen'] = total_loss_gen

            self.outputs_train.append(self._losses)
            self.outputs_eval.append(self._losses)


    def _add_optims(self, optims: dict):
        """
        Adds optims to model that are to be used by optimizers or during training

        Parameters
        ----------
        optim: dict
            dictionary containing all optimizers, optimizers should be of Type[tf.train.Optimizer]
        """
        if self._optims is not None and len(optims) != 0:
            logging.warning('Change of optims is not yet supported')
            pass
        elif self._optims is not None and len(optims) == 0:
            pass
        else:
            self._optims = optims

            optim_gen = self._optims['gen']
            grads_gen = optim_gen.compute_gradients(self._losses['total_gen'], var_list=self.gen.trainable_variables)
            step_gen = optim_gen.apply_gradients(grads_gen)

            optim_discr = self._optims['discr']
            grads_discr = optim_discr.compute_gradients(self._losses['total_discr'], var_list=self.discr.trainable_variables)
            step_discr = optim_discr.apply_gradients(grads_discr)

            steps = tf.group([step_gen, step_discr])

            self.outputs_train.append(steps)

    @staticmethod
    def _build_models(output_c_dim, df_dim, gf_dim, latent_dim, f_size, is_batchnorm, is_bicycle):
        """
        builds generator and discriminators

        """

        discr = Discriminator(df_dim, f_size, 3, is_batchnorm, is_bicycle)

        gen = Generator_add_all(output_c_dim, gf_dim, f_size, latent_dim, is_batchnorm, is_bicycle)

        return discr, gen

    @staticmethod
    def closure(model: typing.Type[AbstractTfNetwork], data_dict: dict,
                metrics={}, fold=0, **kwargs):
        """
                closure method to do a single prediction.


                Parameters
                ----------
                model: AbstractTfNetwork
                    AbstractTfNetwork or its child-clases
                data_dict : dict
                    dictionary containing the data
                metrics : dict
                    dict holding the metrics to calculate
                fold : int
                    Current Fold in Crossvalidation (default: 0)
                **kwargs:
                    additional keyword arguments

                Returns
                -------
                dict
                    Metric values (with same keys as input dict metrics)
                dict
                    Loss values (with same keys as those initially passed to model.init).
                    Additionally, a total_loss key is added
                list
                    Arbitrary number of predictions as np.array

                """

        loss_vals = {}
        metric_vals = {}
        image_name_real_fl = "real_images_frontlight"
        image_name_real_bl = "real_images_backlight"
        image_name_fake_fl = "fake_images_frontlight"
        image_name_fake_bl = "fake_images_backlight"

        mask = "segmentation mask"

        input_B = data_dict.pop('data')
        input_A = data_dict.pop('seg')

        inputs = np.concatenate((input_A, input_B), axis=1)

        real_fl = input_B[:, :3, :, :]
        real_bl = input_B[:, 3:, :, :]
        real_bl = np.concatenate([real_bl, real_bl, real_bl], 1)

        if model.training == True:
            z_rand1 = np.random.normal(size=(input_A.shape[0], model.latent_dim),
                                       #loc=0.0, scale=1.0
                                       )
            z_rand2 = np.random.normal(size=(input_A.shape[0], model.latent_dim),
                                       #loc=0.0, scale=1.0
                                       )

            fake_b, fake_b1, D_real, D_fake, losses, *_ = model.run(inputs, z_rand1, z_rand2)

            fake_fl = fake_b[:, :3, :, :]
            fake_bl = fake_b[:, 3:, :, :]
            fake_bl = np.concatenate([fake_bl, fake_bl, fake_bl], 1)


            fake_fl = (fake_fl + 1) / 2
            logging.info({"images": {"images": fake_fl, "name": image_name_fake_fl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})

            fake_bl = (fake_bl + 1) / 2

            logging.info({"images": {"images": fake_bl, "name": image_name_fake_bl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})


        elif model.training == False:
            fake_fl = []
            fake_bl = []

            fake_b = []

            z_hist = []
            image_name_fake_fl = "val_" + str(image_name_fake_fl)
            image_name_fake_bl = "val_" + str(image_name_fake_bl)


            z0 = np.random.normal(size=(input_A.shape[0], model.latent_dim))  # start point
            z1 = np.random.normal(size=(input_A.shape[0], model.latent_dim))  # endpoint

            for i in range(model.n_latent):

                # Spherical interpolation of latent using slerp

                _omega = np.arccos(np.tensordot(z0 / np.linalg.norm(z0), z1 / np.linalg.norm(z1)))
                z_interpolated = slerp(z0, z1, 1 / float(model.n_latent), omega=_omega)  # performs actual interpolation

                fake_b_, fake_b1, D_real, D_fake, losses = model.run(inputs, z0, z_interpolated)
                z0 = z_interpolated
                z_hist.append(z0)

                fake_fl_ = fake_b_[:, :3, :, :]
                fake_fl_ = (fake_fl_ + 1) / 2
                fake_fl.append(fake_fl_[0])

                fake_bl_ = fake_b_[:, 3:, :, :]
                fake_bl_ = np.concatenate([fake_bl_, fake_bl_, fake_bl_], 1)
                fake_bl_ = (fake_bl_ + 1) / 2
                fake_bl.append(fake_bl_[0])

                fake_b.append(fake_b_)

            fake_fl = np.array(fake_fl)
            logging.info({'image_grid': {"image_array": fake_fl, "name": image_name_fake_fl,
                                         "title": "output_image", "env_appendix": "_%02d" % fold, "nrow": 1}})
            fake_bl = np.array(fake_bl)
            logging.info({'image_grid': {"image_array": fake_bl, "name": image_name_fake_bl,
                                         "title": "output_image", "env_appendix": "_%02d" % fold, "nrow": 1}})


            hist_name = "sampled_z"

            logging.info({"histogram": {"array": np.array(z_hist), "name": hist_name,
                                        "title": "output_image", "env_appendix": "_%02d" % fold}})

        for key, loss_val in losses.items():
            loss_vals[key] = loss_val

        if model.training == False:
            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key, metric_fn in metrics.items():
                metric_fl = []
                metric_bl = []
                for i in range(model.n_latent):
                    fl = fake_fl[i].reshape(1, *fake_fl[i].shape)
                    metric_fl.append(metric_fn(real_fl, fl))
                    bl = fake_bl[i].reshape(1, *fake_bl[i].shape)
                    metric_bl.append(metric_fn(real_bl, bl))

                metric_vals[key + '_FL_mean'] = np.mean(np.array(metric_fl))
                metric_vals[key + '_BL_mean'] = np.mean(np.array(metric_bl))


            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

            image_name_real_fl = "val_" + str(image_name_real_fl)
            image_name_real_bl = "val_" + str(image_name_real_bl)

            mask = "val_" + str(mask)

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})
        gt = input_A

        logging.info({"images": {"images": np.concatenate([gt, gt, gt], 1), "name": mask,
                                 "title": "input_image", "env_appendix": "_%02d" % fold}})

        real_fl = (real_fl + 1) / 2

        logging.info({"images": {"images": real_fl, "name": image_name_real_fl,
                                 "title": "input_image", "env_appendix": "_%02d" % fold}})

        real_bl = (real_bl + 1) / 2

        logging.info({"images": {"images": real_bl, "name": image_name_real_bl,
                                 "title": "input_image", "env_appendix": "_%02d" % fold}})

        if model.training == True:
            return metric_vals, loss_vals, [fake_b, fake_b1, D_real, D_fake]

        else:

            return metric_vals, loss_vals, [fake_b]