import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

class EncoderBlock(layers.Layer):
    """
    Encoder block that takes as input a time series and a numerial representation of a county name
    and creates a learned representation to be processed further in the model.
    """
    def __init__(self, rnn_units, rnn_layer, rnn_activation, pad_val, l1=0, l2=0, dropout=0):
        super(EncoderBlock, self).__init__()
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

        self.mask_layer = None
        if pad_val:
            self.mask_layer = layers.Masking(mask_value=pad_val)

        self.hidden_rnn = rnn_layer(rnn_units, activation=rnn_activation, kernel_regularizer=regularizer, dropout=dropout,
                                    name="rnn_encoder")
        self.hidden_dense = layers.Dense(1, name="name_encoder")

    def call(self, inputs):
        mask = None
        if self.mask_layer:
            masked_inputs = self.mask_layer(inputs[0])

        h_rnn = self.hidden_rnn(masked_inputs)
        h_dense = self.hidden_dense(inputs[1])

        return layers.concatenate([h_rnn, h_dense], name="context")

class RNNMultiOutputIndividual(keras.Model):
    """
    Multi output RNN model with individual weights on the output nodes.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation, l1, l2, dropout, pad_val=None):
        super(RNNMultiOutputIndividual, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation, pad_val, l1, l2, dropout)

        self.c_out = layers.Dense(output_size, name="confirmed")
        self.d_out = layers.Dense(output_size, name="deceased")
        self.r_out = layers.Dense(output_size, name="recovered")

    def call(self, inputs):
        context = self.encoder(inputs)
        return self.c_out(context), self.d_out(context), self.r_out(context)

class RNNMultiOutputShared(keras.Model):
    """
    Multi output RNN model with shared weights on the output nodes.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation, pad_val=None):
        super(RNNMultiOutputShared, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation, pad_val)

        self.rep_vec = layers.RepeatVector(output_size)

        self.c_out = layers.TimeDistributed(layers.Dense(1), name="confirmed")
        self.d_out = layers.TimeDistributed(layers.Dense(1), name="deceased")
        self.r_out = layers.TimeDistributed(layers.Dense(1), name="recovered")

    def call(self, inputs):
        context = self.encoder(inputs)
        context = self.rep_vec(context)
        return self.c_out(context), self.d_out(context), self.r_out(context)

class RNNSingleOutput(keras.Model):
    """
    Single output RNN model with shared weights on the output node.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation, pad_val=None):
        super(RNNSingleOutput, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation, pad_val)

        self.rep_vec = layers.RepeatVector(output_size)

        self.output_node = layers.TimeDistributed(layers.Dense(3))  # 3 is the number of features in the data.

    def call(self, inputs):
        context = self.encoder(inputs)
        context = self.rep_vec(context)
        return self.output_node(context)

class RNNSingleOutputQuantile(keras.Model):
    """
    One output node for each quantile. Each output node produces all features.
    e.g. 3 quantile and 3 features = 3 outputs (each has 3 features), one for each quantile.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation, pad_val=None):
        super(RNNSingleOutputQuantile, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation, pad_val)

        self.rep_vec = layers.RepeatVector(output_size)

        # 3 is the number of features in the data.
        self.output_node_q1 = layers.TimeDistributed(layers.Dense(3), name="output_q0.05")
        self.output_node_q2 = layers.TimeDistributed(layers.Dense(3), name="output_q0.5")
        self.output_node_q3 = layers.TimeDistributed(layers.Dense(3), name="output_q0.95")

    def call(self, inputs):
        context = self.encoder(inputs)
        context = self.rep_vec(context)
        return self.output_node_q1(context), self.output_node_q2(context), self.output_node_q3(context)

class RNNMultiOutputQuantile(keras.Model):
    """
    Each output node produces the values for a featue at a quantile.
    e.g. 3 quantile and 3 features = 9 outputs.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation, pad_val=None):
        super(RNNMultiOutputQuantile, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation, pad_val)

        self.c_out_q1 = layers.Dense(output_size, name="confirmed_q1")
        self.c_out_q2 = layers.Dense(output_size, name="confirmed_q2")
        self.c_out_q3 = layers.Dense(output_size, name="confirmed_q3")

        self.d_out_q1 = layers.Dense(output_size, name="deceased_q1")
        self.d_out_q2 = layers.Dense(output_size, name="deceased_q2")
        self.d_out_q3 = layers.Dense(output_size, name="deceased_q3")

        self.r_out_q1 = layers.Dense(output_size, name="recovered_q1")
        self.r_out_q2 = layers.Dense(output_size, name="recovered_q2")
        self.r_out_q3 = layers.Dense(output_size, name="recovered_q3")

    def call(self, inputs):
        context = self.encoder(inputs)
        return (self.c_out_q1(context), self.c_out_q2(context), self.c_out_q3(context),
    self.d_out_q1(context), self.d_out_q2(context), self.d_out_q3(context),
    self.r_out_q1(context), self.r_out_q2(context), self.r_out_q3(context))

def compileModel(my_model, optimizer, loss, metrics):
    my_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def fitModel(my_model, x, y, epochs, callbacks=None, verbose=2):
    return my_model.fit(x=x, y=y, epochs=epochs, callbacks=callbacks, verbose=verbose)

def evaluateModel(my_model, x, y, verbose=1, return_dict=True):
    return my_model.evaluate(x=x, y=y, verbose=verbose, return_dict=return_dict)
