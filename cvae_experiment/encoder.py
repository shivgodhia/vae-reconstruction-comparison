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
        """Trains the model on data generated batch-by-batch by a Python generator (or an instance of Sequence).
        
        :param train_generator: A generator in order to avoid duplicate data when using multiprocessing. The output of the generator must be either
        - a tuple (inputs, targets)
        - a tuple (inputs, targets, sample_weights).
        This tuple (a single output of the generator) makes a single batch. Therefore, all arrays in this tuple must have the same length (equal to the size of this batch). Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size. The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
        :type train_generator: generator
        :param test_generator: Generates test data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
        :type test_generator: generator
        :param n_samples: number of samples
        :type n_samples: int
        :param batch_size: Number of samples per gradient update
        :type batch_size: int
        :param epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y provided. Note that epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
        :type epochs: int
        :param validation_steps: Total number of steps (batches of samples) to validate before stopping.
        :type validation_steps: int
        """
        self._model.fit_generator(train_generator,
                                  epochs=epochs,
                                  steps_per_epoch=math.ceil(n_samples /batch_size),
                                  validation_data=test_generator,
                                  validation_steps=validation_steps,
                                  **kwargs)

    def predict(self, X):
        """Generates output predictions for the input samples.
        
        :param X: input samples
        :type X: numpy.ndarray (or list of Numpy arrays if the model has multiple inputs)
        :return: predictions
        :rtype: numpy.ndarray
        """
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
