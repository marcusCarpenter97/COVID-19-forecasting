import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Removes tensorflow messages.

import numpy as np
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Activation


class myLSTM:

    def __init__(self):
        self.model = None
        self.history = None

    def create_seq_simple_LSTM(self, nodes=10, in_shape=(4,1), out_shape=1, loss='mean_squared_error', opti='adam'):
        self.model = Sequential()
        self.model.add(LSTM(nodes, input_shape=in_shape))
        self.model.add(Dense(out_shape))
        self.model.compile(loss=loss, optimizer=opti)

    def train(self, x_train, y_train, e=50, b_size=1, v=0):
        self.history = self.model.fit(x_train, y_train, epochs=e, batch_size=b_size, verbose=v)

    def predict(self, data):
        return self.model.predict(data)

    def rmse(self, target, prediction):
        return np.sqrt(((target - prediction) ** 2).mean())
