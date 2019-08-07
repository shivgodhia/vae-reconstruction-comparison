from keras.callbacks import Callback, EarlyStopping
from keras.models import load_model
import numpy as np
import math

TRAIN_LOSS_KEY = 'loss'
VAL_LOSS_KEY = 'val_loss'


class Encoder(object):

    def post_init(self):
        print(f'Initialised {self.__class__.__name__}:')
        print(self._model.summary())

    def fit(self, X_train, X_test, epochs=10,
            batch_size=1000, patience=5, callbacks=[],
            **kwargs):
            """Trains the model for a given number of epochs (iterations on a dataset).
            
            :param X_train: training data. Note that this is also the target because this class describes an autoencoder
            :type X_train: numpy.ndarray
            :param X_test: test data, to be used as validation data
            :type X_test: numpy.ndarray
            :param epochs: Number of epochs to train the model, defaults to 10. An epoch is an iteration over the entire x and y provided. Note that epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
            :type epochs: int, optional
            :param batch_size: Number of samples per gradient update, defaults to 1000
            :type batch_size: int, optional
            :param patience: number of epochs with no improvement after which training will be stopped, defaults to 5
            :type patience: int, optional
            :param callbacks: List of callbacks to apply during training and validation, defaults to []
            :type callbacks: list of keras.callbacks.Callback instances, optional
            """
        early_stop = EarlyStopping(monitor=VAL_LOSS_KEY, min_delta=0,
                                   patience=patience, verbose=1)
        callbacks.append(early_stop)
        assert all([isinstance(item, Callback) for item in callbacks])
        self._model.fit(X_train, X_train, epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, X_test),
                        callbacks=callbacks, verbose=1,
                        **kwargs)

    def fit_generator(self, train_generator, test_generator,
                      n_samples, batch_size, epochs, validation_steps,
                      **kwargs):
        self._model.fit_generator(train_generator,
                                  epochs=epochs,
                                  steps_per_epoch=math.ceil(n_samples /
                                                            batch_size),
                                  validation_data=test_generator,
                                  validation_steps=validation_steps,
                                  **kwargs)

    def predict(self, X):
        return self._model.predict(X)

    def encode(self, X):
        return self._encoder.predict(X)

    def save(self, path):
        self._model.save(f'{path}.h5')
        print(f'--> Model exported to: {path}.h5')

    def load(self, path):
        self._model = load_model(path)

    @property
    def training_loss_history(self):
        return np.array(self._model.history.history[TRAIN_LOSS_KEY])

    @property
    def validation_loss_history(self):
        return np.array(self._model.history.history[VAL_LOSS_KEY])
