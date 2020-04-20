import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes tensorflow messages.

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.callbacks.callbacks import EarlyStopping
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class myLSTM:

    def __init__(self):
        self.model = None
        self.history = None
        self.train_rmse = None
        self.test_rmse = None

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

    def train(self, x_train, y_train, e=50, b_size=1, v=0):
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.history = self.model.fit(x_train, y_train, epochs=e, batch_size=b_size, verbose=v, callbacks=[early_stopping])

    def predict(self, data):
        return self.model.predict(data)

    def rmse(self, target, prediction):
        return np.sqrt(((target - prediction) ** 2).mean())

    def swish(self, x, beta=1.0):
            return x * K.sigmoid(beta * x)

    def plot_history(self):
        # Plot model accuracy hostory.
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Plot model loss history.
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
