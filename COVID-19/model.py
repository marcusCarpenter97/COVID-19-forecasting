import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EncoderBlock(layers.Layer):
    """
    Encoder block that takes as input a time series and a numerial representation of a county name
    and creates a learned representation to be processed further in the model.
    """
    def __init__(self, rnn_units, rnn_layer, rnn_activation, pad_val, l1=0, l2=0, dropout=0):
        super().__init__()
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

        self.mask_layer = None
        if pad_val:
            self.mask_layer = layers.Masking(mask_value=pad_val)

        self.hidden_rnn = rnn_layer(rnn_units, activation=rnn_activation, kernel_regularizer=regularizer, dropout=dropout,
                                    name="rnn_encoder")
        self.hidden_dense = layers.Dense(1, name="name_encoder")

    def call(self, inputs):
        if self.mask_layer:
            masked_inputs = self.mask_layer(inputs[0])

        h_rnn = self.hidden_rnn(masked_inputs)
        h_dense = self.hidden_dense(inputs[1])

        return layers.concatenate([h_rnn, h_dense], name="context")

class MultiOutputRNN(keras.Model):
    """
    Multi output RNN model with individual weights on the output nodes.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation, l1, l2, dropout, pad_val=None):
        super().__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation, pad_val, l1, l2, dropout)

        self.c_out = layers.Dense(output_size, name="confirmed")
        self.d_out = layers.Dense(output_size, name="deceased")
        self.r_out = layers.Dense(output_size, name="recovered")

    def call(self, inputs):
        context = self.encoder(inputs)
        return self.c_out(context), self.d_out(context), self.r_out(context)

class Model():
    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x, y, epochs, callbacks=None, verbose=2):
        return self.model.fit(x=x, y=y, epochs=epochs, callbacks=callbacks, verbose=verbose)

    def evaluate(self, x, y, verbose=1, return_dict=True):
        return self.model.evaluate(x=x, y=y, verbose=verbose, return_dict=return_dict)
