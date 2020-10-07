""" Module containing all the global variables and data structures used in the main file. """
import os
import tensorflow as tf
import tensorflow_addons as tfa
import models
import data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduces tensorflow messages.

UNITS = 100
OUTPUT_SIZE = 28

EPOCHS = 300
VERBOSE = 2
EVAL_VERBOSE = 0

EXAMPLE_COUTRIES = ["United Kingdom", "Germany", "France", "Spain", "Italy", "Brazil", "US", "Mexico", "Australia", "Pakistan",
                    "Yemen", "Afghanistan", "China", "India", "Angola", "Nigeria"]

CHECKPOINT_PATHS = {"multiOutIndvLstm": "checkpoints/multiOutIndvLstm.ckpt",
                    "multiOutSharedLstm": "checkpoints/multiOutSharedLstm.ckpt",
                    "multiOutIndvGru": "checkpoints/multiOutIndvGru.ckpt",
                    "multiOutSharedGru": "checkpoints/multiOutSharedGru.ckpt",
                    "singleOutSharedLstm": "checkpoints/singleOutSharedLstm.ckpt",
                    "singleOutSharedGru": "checkpoints/singleOutSharedGru.ckpt",
                    "multiOutQuantLstm": "checkpoints/multiOutQuantLstm.ckpt",
                    "multiOutQuantGru": "checkpoints/multiOutQuantGru.ckpt",
                    "singleOutQuantLstm": "checkpoints/singleOutQuantLstm.ckpt",
                    "singleOutQuantGru": "checkpoints/singleOutQuantGru.ckpt"}

ACTIVATIONS = {"tanh" : tf.keras.activations.tanh,
               "relu" : tf.keras.activations.relu}

RNN_LAYERS = {"lstm" : tf.keras.layers.LSTM,
              "gru" : tf.keras.layers.GRU}

COMPILE_PARAMS = {"optimizer" : tf.keras.optimizers.Adam(),
                  "loss" : tf.keras.losses.MeanSquaredError(),
                  "metrics" : [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]}

QUANTILE_LOSSES = [tfa.losses.PinballLoss(tau=0.05), tfa.losses.PinballLoss(tau=0.5), tfa.losses.PinballLoss(tau=0.95)]

# The multi output quantile model requires a quantile loss for each output node.
MULTI_QUANTILE_LOSSES = [QUANTILE_LOSSES for _  in range(len(QUANTILE_LOSSES))]  # Make copies.
MULTI_QUANTILE_LOSSES = [item for sublist in MULTI_QUANTILE_LOSSES for item in sublist]  # Flatten list of lists.

QUANTILE_METRICS = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(),
                    tfa.losses.PinballLoss(tau=0.05, name="q0.05"),
                    tfa.losses.PinballLoss(tau=0.5, name="q0.5"),
                    tfa.losses.PinballLoss(tau=0.95, name="q0.95")]

# Create models.
MULTI_MODELS = {"multiOutIndvLstm" : models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"], ACTIVATIONS["tanh"]),
                "multiOutSharedLstm" : models.RNNMultiOutputShared(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"], ACTIVATIONS["tanh"]),
                "multiOutIndvGru" : models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"], ACTIVATIONS["tanh"]),
                "multiOutSharedGru" : models.RNNMultiOutputShared(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"], ACTIVATIONS["tanh"])}

SINGLE_MODELS = {"singleOutSharedLstm" : models.RNNSingleOutput(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"], ACTIVATIONS["tanh"]),
                 "singleOutSharedGru" : models.RNNSingleOutput(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"], ACTIVATIONS["tanh"])}

MULTI_QUANTILE_MODELS = {"multiOutQuantLstm" : models.RNNMultiOutputQuantile(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"],
                                                                             ACTIVATIONS["tanh"]),
                         "multiOutQuantGru" : models.RNNMultiOutputQuantile(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"],
                                                                            ACTIVATIONS["tanh"])}

SINGLE_QUANTILE_MODELS = {"singleOutQuantLstm" : models.RNNSingleOutputQuantile(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"],
                                                                                ACTIVATIONS["tanh"]),
                          "singleOutQuantGru" : models.RNNSingleOutputQuantile(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"],
                                                                               ACTIVATIONS["tanh"])}

# Load and prepare data.
COVID_DATA = data.Data()
COVID_DATA.split_train_test(OUTPUT_SIZE)
COVID_DATA.standarize_data()
enc_names = COVID_DATA.encode_names()
train_x, train_y, test_x, test_y = COVID_DATA.retreive_data()
multi_train_y = COVID_DATA.make_multi_output_data(train_y)
multi_test_y = COVID_DATA.make_multi_output_data(test_y)

# Create train and test datasets.
INPUT_TRAIN_DATA = [train_x, enc_names]
OUTPUT_TRAIN_DATA = {"multiOutput" : [multi_train_y[0], multi_train_y[1], multi_train_y[2]],
                     "singleOutput" : [train_y],
                     "multiOutQuant" : [multi_train_y[0], multi_train_y[0], multi_train_y[0],
                                        multi_train_y[1], multi_train_y[1], multi_train_y[1],
                                        multi_train_y[2], multi_train_y[2], multi_train_y[2]],
                     "singleOutQuant" : [train_y, train_y, train_y]}

INPUT_TEST_DATA = [test_x, enc_names]
OUTPUT_TEST_DATA = {"multiOutput" : [multi_test_y[0], multi_test_y[1], multi_test_y[2]],
                    "singleOutput" : [test_y],
                    "multiOutQuant" : [multi_test_y[0], multi_test_y[0], multi_test_y[0],
                                       multi_test_y[1], multi_test_y[1], multi_test_y[1],
                                       multi_test_y[2], multi_test_y[2], multi_test_y[2]],
                    "singleOutQuant" : [test_y, test_y, test_y]}
