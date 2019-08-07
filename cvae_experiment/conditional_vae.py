from keras.layers.core import Dense, Lambda 
from keras.layers import (Input, Dropout, Concatenate,
                          BatchNormalization, Activation)
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error
from keras.metrics import mse as mse_metric, mae as mae_metric
import numpy as np

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


class CVAE(Encoder):

    LOSS_NAME = 'mse + mmd'

    def __init__(self, n_features,
                 n_categories,
                 latent_dim,
                 intermediate_dims,
                 drop_out=0.2,
                 activation='relu',
                 loss='kl',
                 lr=1e-3,):
        """[summary]
        
        :param n_features: [description]
        :type n_features: [type]
        :param n_categories: [description]
        :type n_categories: [type]
        :param latent_dim: [description]
        :type latent_dim: [type]
        :param intermediate_dims: [description]
        :type intermediate_dims: [type]
        :param drop_out: [description], defaults to 0.2
        :type drop_out: float, optional
        :param activation: [description], defaults to 'relu'
        :type activation: str, optional
        :param loss: [description], defaults to 'kl'
        :type loss: str, optional
        :param lr: [description], defaults to 1e-3
        :type lr: [type], optional
        :return: [description]
        :rtype: [type]
        """

        self.latent_dim = latent_dim
        self.n_features = n_features

        # build encoder model
        X = Input(shape=(n_features, ), name='numerical_input')
        cond = Input(shape=(n_categories, ), name='categorical_input')
        inputs = Concatenate(axis=1)([X, cond])

        x = inputs
        for dim in intermediate_dims:
            x = Dense(dim, activation='linear')(x)
            x = Activation(activation)(x)
            x = Dropout(drop_out)(x)

        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # instantiate encoder model
        self._encoder = Model([X, cond], z_mean, name='encoder')
        
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])
        z_cond = Concatenate(axis=1)([z, cond])

        x = z_cond
        for dim in intermediate_dims:
            x = Dense(dim, activation='linear')(x)
            x = Activation(activation)(x)
            x = Dropout(drop_out)(x)
        outputs = Dense(n_features, activation='relu')(x)

        self._model = Model([X, cond], outputs)
        
        # custom metric
        def mean_squared_error(input, output):
            return mse_metric(X, output)
        
        def mean_absolute_error(input, output):
            return mae_metric(X, output)

        def _model_loss(x, x_decoded_mean):
            xent_loss = mean_squared_error(X, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean)
                                     - K.exp(z_log_var))
            true_z = K.random_normal(K.stack([K.shape(z_mean)[0], self.latent_dim]))
            divergence = 0.1 * compute_mmd(true_z, z)
            if loss == 'kl':
                loss_value = K.mean(xent_loss + kl_loss)
            elif loss == 'mmd':
                loss_value = K.mean(xent_loss + divergence)
            return loss_value

        # self._model.compile(optimizer='Adam', loss=_model_loss, metrics=[mse])
        # TODO: need to change the MSE for this to the custom model above
        self._model.compile(optimizer='Adam', loss=_model_loss,
                            metrics=[mean_squared_error, mean_absolute_error])
        self._model.summary()
