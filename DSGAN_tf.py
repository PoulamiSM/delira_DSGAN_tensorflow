import tensorflow as tf
import os
import argparse
from delira.training import Parameters

from delira.models.gan.lpips_score import lpips_score

parser = argparse.ArgumentParser(description='')
parser.add_argument('--image_size', dest='image_size', default=256) # output image size
parser.add_argument('--input_c_dim', dest='input_c_dim', default=1) # input channel dimension
parser.add_argument('--output_c_dim', dest='output_c_dim', default=4) # output channel dimension
parser.add_argument('--df_dim', dest='df_dim', default=64)  # number of filters in the first layer of discriminator
parser.add_argument('--gf_dim', dest='gf_dim', default=64) # number of filters in the first layer of generator
parser.add_argument('--f_size', dest='f_size', default=4)  # size of kernel
parser.add_argument('--latent_dim', dest='latent_dim', default=16) # dimension of the latent space
parser.add_argument('--n_latent', dest='n_latent', default=8) # number of times latent space sampled/ number of modes
parser.add_argument('--coeff_reconstruct', dest='coeff_reconstruct', default=10) # Weighting factor for reconstruction loss
parser.add_argument('--coeff_ds', dest='coeff_ds', default=8) # Weighting factor for DS regularisor term
parser.add_argument('--is_batchnorm', dest='is_batchnorm', default=True) # when False Instance Normalisation applied
parser.add_argument('--val_batch_size', dest='val_batch_size', default=1) # batchsize for validation
parser.add_argument('--batch_size', dest='batch_size', default=4)  # batchsize for training
parser.add_argument('--num_epochs', dest='num_epochs', default=250) # number of epochs to train
parser.add_argument('--learning_rate', dest='learning_rate', default=2e-5)
parser.add_argument('--dataset_name', dest='dataset_name', default='Fabric21') # Training dataset
parser.add_argument('--val_dataset', dest='val_dataset', default='Fabric26') # Validation dataset
parser.add_argument('--exp_name', dest='exp_name', default='issue_14')
parser.add_argument('--file_name', dest='file_name', default='')

args = parser.parse_args()


params = Parameters(fixed_params={
    "model": {
        "image_size": args.image_size,
        "input_c_dim": args.input_c_dim,
        "output_c_dim": args.output_c_dim,
        "df_dim": args.df_dim,
        "gf_dim": args.gf_dim,
        "f_size": args.f_size,
        "latent_dim": int(args.latent_dim),
        "n_latent": int(args.n_latent),
        "coeff_reconstruct": float(args.coeff_reconstruct),
        "coeff_ds": int(args.coeff_ds),
        "is_batchnorm": args.is_batchnorm,
        #"d_layers": args.d_layers
    },
    "training": {
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "num_epochs": args.num_epochs,

        "optimizer_cls": tf.train.AdamOptimizer, # optimization algorithm to use
        "optimizer_params": {'learning_rate': args.learning_rate,
                             'beta1': 0.5,
                             'beta2': 0.999
                            },
        "criterions": {'L1': tf.losses.absolute_difference,
                       'L2': tf.losses.mean_squared_error,
                       },
        "lr_sched_cls": None,  # the learning rate scheduling algorithm to use
        "lr_sched_params": {}, # the corresponding initialization parameters
        "metrics": {'LPIPS': lpips_score([None, 3, args.image_size, args.image_size])}, # evaluation metrics
        "dataset_name": args.dataset_name,
        "val_dataset": args.val_dataset,
        "exp_name": args.exp_name
    }
})


### Path to log the checkpoints is set here

from trixi.logger.tensorboard.tensorboardxlogger import TensorboardXLogger
from delira.logging import TrixiHandler
import logging

exp_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(params.nested_get("dataset_name"),
                                         params.nested_get("val_dataset"), params.nested_get("image_size"),
                                         params.nested_get("batch_size"), params.nested_get("num_epochs"),
                                            params.nested_get("latent_dim"),
                                         params.nested_get("learning_rate"), args.file_name)

log_path = '/work/scratch/poulami/TB_logs/Delira/BicycleGAN'
logger_cls = TensorboardXLogger


logging.basicConfig(level=logging.INFO,
                    handlers=[TrixiHandler(logger_cls, 0, os.path.join(log_path, params.nested_get("exp_name"), exp_name + '/'))])

logger = logging.getLogger("Test Logger")


### Data loading

from delira.data_loading import ConditionalGanDataset
from delira.data_loading.load_fn import load_sample_cgan

root_path = '/work/scratch/poulami/Dataset/LW Data/'
path_train = os.path.join(root_path, params.nested_get("dataset_name") + '/Original50mm/')

"""
train_dataset = ['Fabric7', 'Fabric8', 'Fabric25', 'Fabric26']
path_train = []
for i in train_dataset:
    dataset_name = i
    path_train.append(os.path.join(root_path, dataset_name + '/Original50mm/'))
"""

path_val = os.path.join(root_path, params.nested_get("val_dataset") + '/Original50mm/')
"""
val_dataset = ['Fabric21']
path_val = []
for i in val_dataset:
    dataset_name = i
    path_val.append(os.path.join(root_path, dataset_name + '/Original50mm/'))
"""

dataset_train = ConditionalGanDataset(path_train, load_sample_cgan, ['.PNG', '.png'], ['.PNG', '.png'])


dataset_val = ConditionalGanDataset(path_val, load_sample_cgan, ['.PNG', '.png'], ['.PNG', '.png'])


### Transforms applied to data

from batchgenerators.transforms import RandomCropTransform, Compose
from batchgenerators.transforms.spatial_transforms import ResizeTransform


transforms = Compose([
    ResizeTransform((1024, 1024), order=1),
    RandomCropTransform((params.nested_get("image_size"), params.nested_get("image_size"))), # Perform Random Crops of Size 256 x 256 pixels,
    ])



###  Data manager

from delira.data_loading import BaseDataManager, SequentialSampler, RandomSampler


manager_train = BaseDataManager(dataset_train, params.nested_get("batch_size"),
                                transforms=transforms,
                                sampler_cls=RandomSampler,
                                n_process_augmentation=4)

manager_val = BaseDataManager(dataset_val, params.nested_get("val_batch_size"),
                              transforms=transforms,
                              sampler_cls=SequentialSampler,
                              n_process_augmentation=4)

import warnings
warnings.simplefilter("ignore", UserWarning) # ignore UserWarnings raised by dependency code
warnings.simplefilter("ignore", FutureWarning) # ignore FutureWarnings raised by dependency code



### Experiment setup for training

from delira.training import TfExperiment
from delira.training.train_utils import create_optims_gan_default_tf, create_optims_bicyclegan_default_tf
from delira.models.gan import DSGenerativeAdversarialNetworkBaseTf



experiment = TfExperiment(params, DSGenerativeAdversarialNetworkBaseTf,
                          name=exp_name,
                          save_path=os.path.join("/work/scratch/poulami/Delira_experiments/DSGAN",
                                                  params.nested_get("exp_name")),
                          optim_builder=create_optims_gan_default_tf,
                          gpu_ids=[0], val_score_key='val_LPIPS_BL_mean', val_score_mode='highest')


model = experiment.run(manager_train, manager_val)