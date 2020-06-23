import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes tensorflow messages.

import numpy as np
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.utils import plot_model
from keras.layers.merge import concatenate

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Activation, Dropout, Embedding, Flatten, RepeatVector, TimeDistributed
from keras.callbacks.callbacks import EarlyStopping
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class myLSTM:

    def __init__(self):
        get_custom_objects().update({'swish': Activation(self.swish)})

        self.model = None
        self.history = None
        self.train_predictions = None
        self.test_predictions = None
        self.hyper_params = None

    def multivariate_embedded_lstm(self, word_size, embed_size, vocab_size, lstm_shape, out_shape,
                           nodes=10, loss='mean_squared_error', optimizer='adam', dropout=0.1,
                           lstm_activation='relu', dense_activation='relu'):
        """
        Parameters
        ----------
        word_size : int
            Size of word verctor used to represent country names.
        embed_size : int
            Size of the output from the embeding layer.
        vocab_size : int
            Number of unique words in the vocabulary.
        lstm_shape : tuple
            A tuple representing the shape of the input for the LSTM layer.
        out_shape : int
            Number of outputs to be produced.
        """
        # Embedding layer for the coutry names.
        embedded_input = Input([word_size])
        embedding_layer = Embedding(vocab_size, embed_size, input_length=word_size)(embedded_input)
        flat_embedded_layer = Flatten()(embedding_layer)

        # Two layer LSTM for the time series.
        lstm_input = Input(lstm_shape)
        lstm_layer = LSTM(nodes, activation=lstm_activation, dropout=dropout)(lstm_input)

        # Merge both.
        merged = concatenate([flat_embedded_layer, lstm_layer])

        # Dense output layer to make predictions.
        output_layer = Dense(out_shape, activation=dense_activation)(merged)

        # Create model from the layers.
        self.model = Model(inputs=[embedded_input, lstm_input], outputs=output_layer, name="COVID-19_Multivariate_LSTM")
        self.model.compile(loss=loss, optimizer=optimizer)

    def train(self, x_train, y_train, epochs=500, batch_size=32, verbose=0, patience=10):

        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        self.history = self.model.fit(x_train, y_train, epochs=epochs,
                                      batch_size=batch_size, verbose=verbose,
                                      callbacks=[early_stopping],
                                      validation_split=0.2) 

    def predict(self, data):
        return self.model.predict(data)

    def make_predictions(self, country):
        """
        Create predictions for a country on a weekly basis.

        Parameter:
        country: country obj.

        Returns:
        numpy array containing predictions for country.
        """
        predictions = []
        # Make a prediction for each week in the test data.
        # The last week won't have a ground truth.
        for week in country.test:
            week = week.reshape(1, week.shape[0], week.shape[1])
            predictions.append(self.predict(week))

        predictions = np.stack(predictions)
        # Stacking the arrays produces a 4D array which needs to be reshaped to 2D.
        return predictions.reshape(predictions.shape[0] * predictions.shape[2], predictions.shape[3])

    def rmse(self, prediction, target):
        return np.sqrt(((prediction - target) ** 2).mean())

    def rmsle(self, prediction, target):
        return np.sqrt(((np.log(prediction+1) - np.log(target+1)) ** 2).mean())

    def mase(self, prediction, target, train_size):
        mean_error = np.mean(np.abs(prediction - target))
        scaling_factor = (1/(train_size-1)) * np.sum(np.abs(np.diff(target.ravel())))
        return mean_error / scaling_factor

    def swish(self, x, beta=1.0):
        return x * K.sigmoid(beta * x)

    def print_summary(self):
        print(self.model.summary())

    def plot_model(self):
        return plot_model(self.model, "model_shape_info.png", show_shapes=True)

    def plot_history(self, fig_name):
        # Plot model loss history.
        fig, ax = plt.subplots()
        ax.plot(self.history.history['loss'])
        ax.plot(self.history.history['val_loss'])
        ax.set_title('model loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'test'], loc='best')
        fig.savefig(f"hist_{fig_name}")
