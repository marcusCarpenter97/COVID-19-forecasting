import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

class EncoderBlock(layers.Layer):
    """
    Encoder block that takes as input a time series and a numerial representation of a county name
    and creates a learned representation to be processed further in the model.
    """
    def __init__(self, rnn_units, rnn_layer, rnn_activation):
        super(EncoderBlock, self).__init__()
        self.hidden_rnn = rnn_layer(rnn_units, activation=rnn_activation, name="rnn_encoder")
        self.hidden_dense = layers.Dense(1, name="name_encoder")

    def call(self, inputs):
        h_rnn = self.hidden_rnn(inputs[0])
        h_dense = self.hidden_dense(inputs[1])

        return layers.concatenate([h_rnn, h_dense], name="context")

class RNNMultiOutputIndividual(keras.Model):
    """
    Multi output RNN model with individual weights on the output nodes.
    """
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation):
        super(RNNMultiOutputIndividual, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation)

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
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation):
        super(RNNMultiOutputShared, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation)

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
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation):
        super(RNNSingleOutput, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation)

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
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation):
        super(RNNSingleOutputQuantile, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation)

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
    def __init__(self, output_size, rnn_units, rnn_layer, rnn_activation):
        super(RNNMultiOutputQuantile, self).__init__()
        self.encoder = EncoderBlock(rnn_units, rnn_layer, rnn_activation)

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

def fitModel(my_model, x, y, epochs, verbose=2):
    return my_model.fit(x=x, y=y, epochs=epochs, verbose=verbose)

def evaluateModel(my_model, x, y, verbose=1, return_dict=True):
    return my_model.evaluate(x=x, y=y, verbose=verbose, return_dict=return_dict)

# TODO remove.
def LSTMMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "LSTM"
    layer = layers.LSTM
    return RNNMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def LSTMSingleOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "LSTM"
    layer = layers.LSTM
    return RNNSingleOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def LSTMMultiOutput_V2(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "LSTM"
    layer = layers.LSTM
    return RNNMultiOutput_V2(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def LSTMMultiOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "LSTM"
    layer = layers.LSTM
    return RNNMultiOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def LSTMSingleOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "LSTM"
    layer = layers.LSTM
    return RNNSingleOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def GRUMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "GRU"
    layer = layers.GRU
    return RNNMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def GRUSingleOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "GRU"
    layer = layers.GRU
    return RNNSingleOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def GRUMultiOutput_V2(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "GRU"
    layer = layers.GRU
    return RNNMultiOutput_V2(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def GRUMultiOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "GRU"
    layer = layers.GRU
    return RNNMultiOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def GRUSingleOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "GRU"
    layer = layers.GRU
    return RNNSingleOutputQuantile(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)
