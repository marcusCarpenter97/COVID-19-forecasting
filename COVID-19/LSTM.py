import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes tensorflow messages.

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Activation, Dropout
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

    def create_multivariate_LSTM(self, in_shape, out_shape, nodes=10, loss='mean_squared_error', opti='adam',
            dropout_val=0.1, lstm_activation='tanh', dense_activation='sigmoid'):

        input_layer = keras.Input(in_shape)

        lstm_layer_1 = layers.LSTM(nodes, return_sequences=True, activation=lstm_activation, dropout=dropout_val)(input_layer)

        lstm_layer_2 = layers.LSTM(nodes, activation=lstm_activation, dropout=dropout_val)(lstm_layer_1)

        output_layer = layers.Dense(out_shape, activation=dense_activation)(lstm_layer_2)

        self.model = keras.Model(inputs=input_layer, outputs=output_layer, name="COVID-19_Multivariate_LSTM")

        self.model.compile(loss=loss, optimizer=opti)

    def train(self, x_train, y_train, e, b_size=1, v=0, p=5):
        early_stopping = EarlyStopping(patience=p, restore_best_weights=True)
        self.history = self.model.fit(x_train, y_train, epochs=e, batch_size=b_size, verbose=v, callbacks=[early_stopping],
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

    def print_summmary(self):
        print(self.model.summary())

    def plot_model(self):
        return keras.utils.plot_model(self.model, "model_shape_info.png", show_shapes=True)

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
