from keras.layers.core import Dense, Lambda
from keras.layers import (Input, Dropout, BatchNormalization, Activation)
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error
import numpy as np
from keras import metrics
from encoder import Encoder


TRAIN_LOSS_KEY = 'loss'
VAL_LOSS_KEY = 'val_loss'


def compute_kernel(x, y):
    """Implementation from Shengjia Zhao MMD Variational Autoencoder 
    https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd__model.py#L62 """
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
    tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, dtype='float32'))


def compute_mmd(x, y):
    """Implementation from Shengjia Zhao MMD Variational Autoencoder 
    https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd__model.py#L62 """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


class VAE(Encoder):

    LOSS_NAME = 'mse + mmd'

    def __init__(self, n_features,
                 latent_dim,
                 intermediate_dims,
                 drop_out=0.2,
                 activation='relu',
                 loss='kl',
                 lr=1e-3,):

        self.latent_dim = latent_dim
        self.n_features = n_features

        # build encoder model
        inputs = Input(shape=(n_features,), name='encoder_input')
        x = inputs
        for dim in intermediate_dims:
            x = Dense(dim, activation='linear')(x)
            x = Activation(activation)(x)
            x = Dropout(drop_out)(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # instantiate encoder model
        self._encoder = Model(inputs, z_mean, name='encoder')

        def sampling(args):
            
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])

        x = z
        intermediate_dims.reverse()
        for dim in intermediate_dims:
            x = Dense(dim, activation='linear')(x)
            x = Activation(activation)(x)
            x = Dropout(drop_out)(x)
        ouputs = Dense(n_features, activation='relu')(x)

        self._model = Model(inputs, ouputs)

        def _model_loss(x, x_decoded_mean):
            xent_loss = mean_squared_error(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            true_z = K.random_normal(K.stack([K.shape(z_mean)[0], self.latent_dim]))
            divergence = 0.1 * compute_mmd(true_z, z)
            if loss == 'kl':
                loss_value = K.mean(xent_loss + kl_loss)
            elif loss == 'mmd':
                loss_value = K.mean(xent_loss + divergence)
            return loss_value

        self._model.compile(
            optimizer='Adam', loss=_model_loss, metrics=[metrics.mse, metrics.mae])
        self._model.summary()
