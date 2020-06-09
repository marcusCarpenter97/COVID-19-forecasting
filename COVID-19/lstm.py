import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes tensorflow messages.

import numpy as np
import matplotlib.pyplot as plt

#from tensorflow import keras
from keras import Input, Model
from keras.utils import plot_model
from keras.layers.merge import concatenate

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Activation, Dropout, Embedding, Flatten
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

    def multivariate_encoder_decoder(self, input_shape, out_shape, nodes=200, 
                           loss='mean_squared_error', optimizer='adam', 
                           dropout=0.1, lstm_activation='relu',
                           dense_activation='relu'):

        self.model = Sequential()
        self.model.add(LSTM(nodes, activation=lstm_activation, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(nodes, activation=dense_activation))
        self.model.add(Dense(out_shape))

        self.model.compile(loss=loss, optimizer=optimizer)

    # TODO remove
    def create_simple_LSTM(self, nodes=10, in_shape=(4,1), out_shape=1, loss='mean_squared_error', opti='adam', dropout=0.1,
            lstm_activation='tanh', dense_activation='sigmoid'):
        if lstm_activation == 'swish':
            lstm_activation = self.swish
        if dense_activation == 'swish': 
            dense_activation = self.swish

        self.model = Sequential()
        self.model.add(LSTM(nodes, input_shape=in_shape))
        self.model.add(Activation(lstm_activation))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(out_shape))
        self.model.add(Activation(dense_activation))
        self.model.compile(loss=loss, optimizer=opti)

    # TODO remove
    def create_multivariate_LSTM(self, in_shape, out_shape, vocab_size, output_size, input_size, nodes=10,
            loss='mean_squared_error', opti='adam', dropout_val=0.1, lstm_activation='tanh', dense_activation='sigmoid'):

        # Embedding layer for the coutry names.
        embedded_input = Input([input_size])
        embedding_layer = Embedding(vocab_size, output_size, input_length=input_size)(embedded_input)
        dense_embedding = Dense(nodes)(embedding_layer)
        flat_embedded_layer = Flatten()(dense_embedding)

        # Two layer LSTM for the time series.
        input_layer = Input(in_shape)
        lstm_layer_1 = LSTM(nodes, return_sequences=True, activation=lstm_activation, dropout=dropout_val)(input_layer)
        lstm_layer_2 = LSTM(nodes, activation=lstm_activation, dropout=dropout_val)(lstm_layer_1)

        # Merge both.
        merged = concatenate([flat_embedded_layer, lstm_layer_2])

        # Dense output layer to make predictions.
        output_layer = Dense(out_shape, activation=dense_activation)(merged)

        # Create model from the layers.
        self.model = Model(inputs=[embedded_input, input_layer], outputs=output_layer, name="COVID-19_Multivariate_LSTM")
        self.model.compile(loss=loss, optimizer=opti)

    def train(self, x_train, y_train, epochs=500, batch_size=32, verbose=0, patience=10):

        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        self.history = self.model.fit(x_train, y_train, epochs=epochs,
                                      batch_size=batch_size, verbose=verbose,
                                      callbacks=[early_stopping],
                                      validation_split=0.2) 

    def predict(self, data):
        return self.model.predict(data)

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

    # TODO too specific. Improve
    def plot_predictions(self, fig_name, original, forecast_horizon):
        # The first sample is lost after each differencing, so the + 2 is required. 
        empty_arr = np.empty((forecast_horizon+2, 1))
        empty_arr[:] = np.nan
        shifted_train = np.concatenate([empty_arr, self.train_predictions])
        # The test data mus be shifted by 2 empty arrays plus the training data.
        empty_arr = np.empty(((forecast_horizon+2)*2+len(self.train_predictions), 1))
        empty_arr[:] = np.nan
        shifted_test = np.concatenate([empty_arr, self.test_predictions])

        fig, ax = plt.subplots()
        ax.plot(original)
        ax.plot(shifted_train)
        ax.plot(shifted_test)
        ax.set_title('Prediction over original')
        ax.set_ylabel('Number of infected')
        ax.set_xlabel('Time')
        ax.legend(['original data', 'train predictions', 'test predictions'], loc='best')
        fig.savefig(f"pred_{fig_name}")
