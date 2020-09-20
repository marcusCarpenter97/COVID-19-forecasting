import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def RNNMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation):
    """
    """
    temporal_inputs = keras.Input(shape=temporal_input_shape, name="time_series_input")
    word_inputs = keras.Input(shape=word_input_shape, name="country_name_input")

    hidden_rnn = layer(recurrent_units, activation=activation, name=f"{name}_encoder")(temporal_inputs)
    hidden_dense = layers.Dense(1, name="country_name")(word_inputs)

    context = layers.concatenate([hidden_rnn, hidden_dense], name="context")

    confimed_out = layers.Dense(output_size, name="confirmed")(context)
    deceased_out = layers.Dense(output_size, name="deceased")(context)
    recovered_out = layers.Dense(output_size, name="recovered")(context)

    model = keras.Model(inputs=[temporal_inputs, word_inputs], outputs=[confimed_out, deceased_out, recovered_out], name =
                        f"{name}MultiOutput")

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

def LSTMMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "LSTM"
    layer = layers.LSTM
    return RNNMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def LSTMSingleOutput():
    pass

def GRUMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, activation='relu'):
    name = "GRU"
    layer = layers.GRU
    return RNNMultiOutput(temporal_input_shape, word_input_shape, recurrent_units, output_size, name, layer, activation)

def GRUSingleOutput():
    pass
