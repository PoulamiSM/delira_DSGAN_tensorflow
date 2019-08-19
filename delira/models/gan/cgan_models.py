import tensorflow as tf
import numpy as np

conv2d = tf.keras.layers.Conv2D
maxpool2d = tf.keras.layers.MaxPool2D
dense = tf.keras.layers.Dense
relu = tf.keras.layers.ReLU
gap2d = tf.keras.layers.GlobalAveragePooling2D
batchnorm2d = tf.keras.layers.BatchNormalization
lrelu = tf.keras.layers.LeakyReLU
add = tf.keras.layers.Add
dropout = tf.keras.layers.Dropout
image_format = tf.keras.backend.image_data_format()
instance_norm = tf.keras.layers.LayerNormalization


from SN import spectral




spec = spectral()



def get_image_format_and_axis():
    """
    helper function to read out keras image_format and convert to axis
    dimension
    Returns
    -------
    str
        image data format (either "channels_first" or "channels_last")
    int
        integer corresponding to the channel_axis (either 1 or -1)
    """
    image_format = tf.keras.backend.image_data_format()
    if image_format == "channels_first":
        return image_format, 1
    elif image_format == "channels_last":
        return image_format, -1
    else:
        raise RuntimeError(
                "Image format unknown, got: {}".format(image_format)
                )


class ResBlock(tf.keras.Model):
    def __init__(self, filters_in: int, filters: int,
                 strides: tuple, kernel_size: int, bias=False, is_batchnorm=True):
        super(ResBlock, self).__init__()

        _, _axis = get_image_format_and_axis()
        self.is_batchnorm = is_batchnorm

        self.identity = None
        if filters_in != filters:
            self.identity = conv2d(
                                filters=filters, strides=strides[0],
                                kernel_size=1, padding='same', use_bias=bias,
                                )
            if self.is_batchnorm:
                self.bnorm_identity = batchnorm2d(axis=_axis)
            else:
                self.bnorm_identity = instance_norm(axis=[0, 1])

        self.conv_1 = conv2d(
                        filters=filters, strides=strides[0],
                        kernel_size=kernel_size,
                        padding='same', use_bias=bias,
                        )
        if self.is_batchnorm:
            self.batchnorm_1 = batchnorm2d(axis=_axis)
        else:
            self.instancenorm_1 = instance_norm(axis=[0, 1])

        self.conv_2 = conv2d(
                        filters=filters, strides=strides[1],
                        kernel_size=kernel_size,
                        padding='same', use_bias=bias,
                        )
        if self.is_batchnorm:
            self.batchnorm_2 = batchnorm2d(axis=_axis)
        else:
            self.instancenorm_2 = instance_norm(axis=[0, 1])

        self.relu = relu()
        self.add = add()

    def call(self, inputs, training=None):

        if self.identity:
            identity = self.identity(inputs)
            identity = self.bnorm_identity(identity, training=training)
        else:
            identity = inputs

        x = self.conv_1(inputs)
        if self.is_batchnorm:
            x = self.batchnorm_1(x, training=training)
        else:
            x = self.instancenorm_1(x, training=training)
        x = self.relu(x)
        x = self.conv_2(x)
        if self.is_batchnorm:
            x = self.batchnorm_2(x, training=training)
        else:
            x = self.instancenorm_2(x, training=training)
        x = self.add([x, identity])
        x = self.relu(x)

        return x


class ResNet18(tf.keras.Model):
    """
        Encoder used to encode ground-truth image into latent space values in BicycleGAN
    """
    def __init__(self, df_dim, latent_dim, f_size, is_batchnorm=True, bias=False):
        super(ResNet18, self).__init__()

        _image_format, _axis = get_image_format_and_axis()

        self.is_batchnorm = is_batchnorm
        self.latent_dim = latent_dim

        self.conv1 = conv2d(filters=df_dim, strides=2, kernel_size=3,
                            padding='same', use_bias=bias,
                            )
        if self.is_batchnorm:
            self.batchnorm = batchnorm2d(axis=_axis)
        else:
            self.instancenorm = instance_norm(axis=[0, 1])

        self.relu = relu()
        self.pool1 = maxpool2d(pool_size=3, strides=2, data_format=_image_format)

        self.block_2_1 = ResBlock(filters_in=df_dim, filters=df_dim,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_2_2 = ResBlock(filters_in=df_dim, filters=df_dim,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_3_1 = ResBlock(filters_in=df_dim, filters=df_dim * 2,
                                  strides=(2, 1), kernel_size=3,
                                  bias=bias)

        self.block_3_2 = ResBlock(filters_in=df_dim * 2, filters=df_dim * 2,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_4_1 = ResBlock(filters_in=df_dim * 2, filters=df_dim * 4,
                                  strides=(2, 1), kernel_size=3,
                                  bias=bias)

        self.block_4_2 = ResBlock(filters_in=df_dim * 4, filters=df_dim * 4,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)

        self.block_5_1 = ResBlock(filters_in=df_dim * 4, filters=df_dim * 8,
                                  strides=(2, 1), kernel_size=3,
                                  bias=bias)

        self.block_5_2 = ResBlock(filters_in=df_dim * 8, filters=df_dim * 8,
                                  strides=(1, 1), kernel_size=3,
                                  bias=bias)
        self.dense1 = dense(self.latent_dim)
        self.dense2 = dense(self.latent_dim)
        self.gap = gap2d(data_format=_image_format)

    def call(self, inputs, training=None):

        x = self.conv1(inputs)

        if self.is_batchnorm:
            x = self.batchnorm(x, training=training)
        else:
            x = self.instancenorm(x, training=training)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.block_2_1(x, training=training)
        x = self.block_2_2(x, training=training)

        x = self.block_3_1(x, training=training)
        x = self.block_3_2(x, training=training)

        x = self.block_4_1(x, training=training)
        x = self.block_4_2(x, training=training)

        x = self.block_5_1(x, training=training)
        x = self.block_5_2(x, training=training)

        x = self.gap(x)
        mu = self.dense1(x)
        log_sigma = self.dense2(x)

        z = mu + tf.random_normal(shape=tf.shape(self.latent_dim)) * tf.exp(log_sigma)

        return z, mu, log_sigma


class DiscDownsample(tf.keras.Model):

    def __init__(self, filters, size, strides=2, is_batchnorm=True, apply_norm=True):
        super(DiscDownsample, self).__init__()
        self.apply_norm = apply_norm
        self.is_batchnorm = is_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = conv2d(filters, (size, size), strides=strides, padding='same', kernel_initializer=initializer,
                                        use_bias=False, kernel_constraint=spec)

        if self.apply_norm:
            if self.is_batchnorm:
                self.batchnorm = batchnorm2d()
            else:
                self.instancenorm = instance_norm(axis=[0, 1])
        self.lrelu = lrelu()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_norm:
            if self.is_batchnorm:
                x = self.batchnorm(x, training=training)
            else:
                x = self.instancenorm(x, training=training)
        x = self.lrelu(x)
        return x


class Discriminator(tf.keras.Model):
    """
        Discriminator used for pix2pix (is_bicycle=False) and BicycleGAN, DSGAN (is_bicycle=True)
        ----
        Calls DiscDownsample Class for downsampling
    """

    def __init__(self, df_dim, f_size, d_layers=3, is_batchnorm=True, is_bicycle=False):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.is_batchnorm = is_batchnorm
        self.is_bicycle = is_bicycle
        self.d_layers = d_layers

        self.down1 = DiscDownsample(df_dim, f_size, 2, is_batchnorm,  False)
        self.down2 = DiscDownsample(df_dim * 2, f_size, 2, is_batchnorm)
        self.down3 = DiscDownsample(df_dim * 4, f_size, 2, is_batchnorm)

        if self.is_bicycle and self.d_layers > 3:                            # To add an extra layer for patch size of 142 x 142
            self.down4 = DiscDownsample(df_dim * 8, f_size, 2, is_batchnorm)


        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(df_dim * 8,
                                           (f_size, f_size),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False,
                                           kernel_constraint=spec
                                           )
        if self.is_batchnorm:
            self.batchnorm1 = batchnorm2d()
        else:
            self.instancenorm1 = instance_norm(axis=[0, 1])

        # shape change from (batch_size, 512, 31, 31) to (batch_size, 1, 30, 30)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = conv2d(1, (f_size, f_size), strides=1)


    def call(self, inp, training=None):

        x = self.down1(inp, training=training)
        x = self.down2(x, training=training)
        x = self.down3(x, training=training)

        if self.is_bicycle and self.d_layers > 3:
            x = self.down4(x, training=training)


        x = self.zero_pad1(x)

        x = self.conv(x)
        if self.is_batchnorm:
            x = self.batchnorm1(x, training=training)
        else:
            x = self.instancenorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)

        x = self.last(x)

        return x



class Downsample(tf.keras.Model):

    def __init__(self, filters, f_size, is_batchnorm=True, apply_norm=True):
        super(Downsample, self).__init__()
        self.apply_norm = apply_norm
        self.is_batchnorm = is_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (f_size, f_size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_norm:
            if self.is_batchnorm:
                self.batchnorm = batchnorm2d()
            else:
                self.instancenorm = instance_norm(axis=[0, 1])
        self.lrelu = lrelu()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_norm:
            if self.is_batchnorm:
                x = self.batchnorm(x, training=training)
            else:
                x = self.instancenorm(x, training=training)
        x = self.lrelu(x)
        return x


class Upsample(tf.keras.Model):

    def __init__(self, filters, f_size, is_batchnorm=True, apply_dropout=False):
        super(Upsample, self).__init__()

        self.is_batchnorm = is_batchnorm
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (f_size, f_size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False,
                                                        )
        if self.is_batchnorm:
            self.batchnorm = batchnorm2d()
        else:
            self.instancenorm = instance_norm(axis=[0, 1])
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        if self.is_batchnorm:
            x = self.batchnorm(x, training=training)
        else:
            x = self.instancenorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=1)
        return x


class Generator(tf.keras.Model):
    """
        For concatenating latent code along with Generator input in BicycleGAN and DSGAN
        ----
        Calls Downsample Class and Upsample Class for downsampling and upsampling respectively
    """

    def __init__(self, output_c_dim, gf_dim, f_size, latent_dim=None, is_batchnorm=True, is_bicycle=False):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.is_bicycle = is_bicycle
        self.latent_dim = latent_dim

        self.down1 = Downsample(gf_dim, f_size, is_batchnorm, apply_norm=False)
        self.down2 = Downsample(gf_dim*2, f_size, is_batchnorm)
        self.down3 = Downsample(gf_dim*4, f_size, is_batchnorm)
        self.down4 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down5 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down6 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down7 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down8 = Downsample(gf_dim*8, f_size, is_batchnorm)

        self.up1 = Upsample(gf_dim*8, f_size, is_batchnorm, apply_dropout=True)
        self.up2 = Upsample(gf_dim*8, f_size, is_batchnorm, apply_dropout=True)
        self.up3 = Upsample(gf_dim*8, f_size, is_batchnorm, apply_dropout=True)
        self.up4 = Upsample(gf_dim*8, f_size, is_batchnorm)
        self.up5 = Upsample(gf_dim*4, f_size, is_batchnorm)
        self.up6 = Upsample(gf_dim*2, f_size, is_batchnorm)
        self.up7 = Upsample(gf_dim, f_size, is_batchnorm)

        self.last = tf.keras.layers.Conv2DTranspose(output_c_dim,
                                                    (f_size, f_size),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)


    def call(self, x, z=None, training=None):


        if z is not None and self.is_bicycle:

            image_size = x.shape[-1]
            z = tf.reshape(z, [tf.shape(x)[0], self.latent_dim, 1, 1])
            z = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x, z], axis=1)
        else:
            G = x

        x1 = self.down1(G, training=training)
        x2 = self.down2(x1, training=training)
        x3 = self.down3(x2, training=training)
        x4 = self.down4(x3, training=training)
        x5 = self.down5(x4, training=training)
        x6 = self.down6(x5, training=training)
        x7 = self.down7(x6, training=training)
        x8 = self.down8(x7, training=training)

        x9 = self.up1(x8, x7, training=training)
        x10 = self.up2(x9, x6, training=training)
        x11 = self.up3(x10, x5, training=training)
        x12 = self.up4(x11, x4, training=training)
        x13 = self.up5(x12, x3, training=training)
        x14 = self.up6(x13, x2, training=training)
        x15 = self.up7(x14, x1, training=training)

        x16 = self.last(x15)
        x16 = tf.nn.tanh(x16)

        return x16

class Generator_add_all(tf.keras.Model):
    """
        Generator used for concatenating latent code along with Generator input as well as to the input of each
         downsampling layer in BicycleGAN and DSGAN

          ----
        Calls Downsample Class and Upsample Class for downsampling and upsampling respectively

    """

    def __init__(self, output_c_dim, gf_dim, f_size, latent_dim=None, is_batchnorm=True, is_bicycle=False):
        super(Generator_add_all, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.is_bicycle = is_bicycle
        self.latent_dim = latent_dim

        self.down1 = Downsample(gf_dim, f_size, is_batchnorm, apply_norm=False)
        self.down2 = Downsample(gf_dim*2, f_size, is_batchnorm)
        self.down3 = Downsample(gf_dim*4, f_size, is_batchnorm)
        self.down4 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down5 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down6 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down7 = Downsample(gf_dim*8, f_size, is_batchnorm)
        self.down8 = Downsample(gf_dim*8, f_size, is_batchnorm)

        self.up1 = Upsample(gf_dim*8, f_size, is_batchnorm, apply_dropout=True)
        self.up2 = Upsample(gf_dim*8, f_size, is_batchnorm, apply_dropout=True)
        self.up3 = Upsample(gf_dim*8, f_size, is_batchnorm, apply_dropout=True)
        self.up4 = Upsample(gf_dim*8, f_size, is_batchnorm)
        self.up5 = Upsample(gf_dim*4, f_size, is_batchnorm)
        self.up6 = Upsample(gf_dim*2, f_size, is_batchnorm)
        self.up7 = Upsample(gf_dim, f_size, is_batchnorm)

        self.last = tf.keras.layers.Conv2DTranspose(output_c_dim,
                                                    (f_size, f_size),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)

    def call(self, x, z=None, training=None):


        if z is not None and self.is_bicycle:

            image_size = x.shape[-1]
            z = tf.reshape(z, [tf.shape(x)[0], self.latent_dim, 1, 1])
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x, z_scaled], axis=1)

            x1 = self.down1(G, training=training)  # (bs, 64, 128, 128)

            image_size = x1.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x1, z_scaled], axis=1)

            x2 = self.down2(G, training=training)  # (bs, 128, 64, 64)

            image_size = x2.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x2, z_scaled], axis=1)

            x3 = self.down3(G, training=training)  # (bs, 256, 32, 32)

            image_size = x3.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x3, z_scaled], axis=1)

            x4 = self.down4(G, training=training)  # (bs, 512, 16, 16)

            image_size = x4.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x4, z_scaled], axis=1)

            x5 = self.down5(G, training=training)  # (bs, 512, 8, 8)

            image_size = x5.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x5, z_scaled], axis=1)

            x6 = self.down6(G, training=training)  # (bs, 512, 4, 4)

            image_size = x6.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x6, z_scaled], axis=1)

            x7 = self.down7(G, training=training)  # (bs, 512, 2, 2)

            image_size = x7.shape[-1]
            z_scaled = tf.tile(z, [1, 1, image_size, image_size])
            G = tf.concat([x7, z_scaled], axis=1)

            x8 = self.down8(G, training=training)  # (bs, 512, 1, 1)

            x9 = self.up1(x8, x7, training=training)  # (bs, 1024, 2, 2)
            x10 = self.up2(x9, x6, training=training)  # (bs, 1024, 4, 4)
            x11 = self.up3(x10, x5, training=training)  # (bs, 1024, 8, 8)
            x12 = self.up4(x11, x4, training=training)  # (bs, 1024, 16, 16)
            x13 = self.up5(x12, x3, training=training)  # (bs, 512, 32, 32)
            x14 = self.up6(x13, x2, training=training)  # (bs, 256, 64, 64)
            x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

            x16 = self.last(x15)  # (bs, 4,  256, 256)
            x16 = tf.nn.tanh(x16)

        return x16


class Discriminator_zz(tf.keras.Model):
    """
        Discriminator between latent space variables to avoid mode collapse.
        Not implemented finally

    """

    def __init__(self):
        super(Discriminator_zz, self).__init__()

        self.add = add()

        self.dense1 = dense(64)
        self.lrelu1 = lrelu(alpha=0.2)
        self.dropout1 = dropout(rate=0.3)

        self.dense2 = dense(128)
        self.lrelu2 = lrelu(alpha=0.2)
        self.dropout2 = dropout(rate=0.3)

        self.dense3 = dense(256)
        self.lrelu3 = lrelu(alpha=0.2)
        self.dropout3 = dropout(rate=0.3)

        self.dense4 = dense(512)
        self.lrelu4 = lrelu(alpha=0.2)
        self.dropout4 = dropout(rate=0.3)

        self.dense = dense(1)

    def call(self, inp1, inp2):

        x = self.add([inp1, inp2])
        x = self.dense1(x)
        x = self.lrelu1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.lrelu2(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.lrelu3(x)
        x = self.dropout3(x)

        x = self.dense4(x)
        x = self.lrelu4(x)
        x = self.dropout4(x)

        x = self.dense(x)

        return x












