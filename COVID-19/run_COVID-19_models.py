import os
import argparse
from pprint import pprint
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import data
import models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduces tensorflow messages.

UNITS = 100
OUTPUT_SIZE = 28

EPOCHS = 300
VERBOSE = 2
EVAL_VERBOSE = 0

EXAMPLE_COUTRIES = ["United Kingdom", "Germany", "France", "Spain", "Italy"]

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

# Get command line options
parser = argparse.ArgumentParser()
parser.add_argument("--load_all", action="store_true", help="Load saved models.")
args = parser.parse_args()

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

# Compile models.
def compile_models(model_dict, losses, metrics):
    for name, model in model_dict.items():
        print(f"Compiling model: {name}")
        models.compileModel(model, COMPILE_PARAMS["optimizer"], losses, metrics)

compile_models(MULTI_MODELS, COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])
compile_models(SINGLE_MODELS, COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])
compile_models(MULTI_QUANTILE_MODELS, MULTI_QUANTILE_LOSSES, QUANTILE_METRICS)
compile_models(SINGLE_QUANTILE_MODELS, QUANTILE_LOSSES, QUANTILE_METRICS)

# Train models.
def train_models(model_dict, out_data):
    models_hist = {}
    for name, model in model_dict.items():
        print(f"Training model: {name}")
        checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATHS[name], save_weights_only=True)
        models_hist[name] = models.fitModel(model, x=INPUT_TRAIN_DATA, y=OUTPUT_TRAIN_DATA[out_data], epochs=EPOCHS,
                                            callbacks=[checkpoint])
    return models_hist

def load_models(models):
    for name, model in models.items():
        model.load_weights(CHECKPOINT_PATHS[name])

if args.load_all:
    print("Loading models...")
    load_models(MULTI_MODELS)
    load_models(SINGLE_MODELS)
    load_models(MULTI_QUANTILE_MODELS)
    load_models(SINGLE_QUANTILE_MODELS)
    print("Done.")
else:
    print("Training models...")
    multi_out_hist = train_models(MULTI_MODELS, "multiOutput")
    single_out_hist = train_models(SINGLE_MODELS, "singleOutput")
    multi_quantile_out_hist = train_models(MULTI_QUANTILE_MODELS, "multiOutQuant")
    single_quantile_out_hist = train_models(SINGLE_QUANTILE_MODELS, "singleOutQuant")
    print("Done.")

# Evaluate models.
def evaluate_models(model_dict, out_data):
    models_eval = {}
    for name, model in model_dict.items():
        print(f"Evaluating model: {name}")
        models_eval[name] = models.evaluateModel(model, x=INPUT_TEST_DATA, y=OUTPUT_TEST_DATA[out_data])
    return models_eval

multi_out_eval = evaluate_models(MULTI_MODELS, "multiOutput")
single_out_eval = evaluate_models(SINGLE_MODELS, "singleOutput")
multi_quantile_out_eval = evaluate_models(MULTI_QUANTILE_MODELS, "multiOutQuant")
single_quantile_out_eval = evaluate_models(SINGLE_QUANTILE_MODELS, "singleOutQuant")

# Print evalutaion results.
def print_eval_res(eval_dict):
    for model_res in eval_dict:
        print(f"Evaluation results for {model_res}")
        pprint(eval_dict[model_res])
        print()

print_eval_res(multi_out_eval)
print_eval_res(single_out_eval)
print_eval_res(multi_quantile_out_eval)
print_eval_res(single_quantile_out_eval)

# Predictions using best model on a selection of countries... (repeat for quantiles)
mulit_out_pred = MULTI_MODELS["multiOutIndvGru"].predict([test_x, enc_names])
mulit_out_pred = np.stack(mulit_out_pred)

# Reshape the predictions to fit the rescaler.
predictions = []
for c, d, r in zip(mulit_out_pred[0], mulit_out_pred[1], mulit_out_pred[2]):
    predictions.append(np.stack([c, d, r]).T)

predictions = np.stack(predictions)

# De-standarize predictions and data...
rescaled_predictions = COVID_DATA.destandarize_data(predictions)

# Plot predictions vs data...
def prepare_country_data_for_plots(name, enc_names, prediction_data):
    country = COVID_DATA.find_country(name)
    country_idx = (enc_names == country.encoded_name).all(axis=1).nonzero()
    country_predictions = prediction_data[country_idx]
    country_predictions = country_predictions.reshape(country_predictions.shape[1], country_predictions.shape[2]).T
    return country_predictions, country.test_y.T

def plot_pred_v_data(pred, og_data, country_name):
    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle(country_name)
    for idx, ax in enumerate(axes):
        ax.plot(pred[idx])
        ax.plot(og_data[idx])
        ax.set_title(sub_titles[idx])
        ax.set_xlabel("Days")
        ax.set_ylabel("People")
        ax.legend(["Model predictions", "Real data"], loc="best")

for country_name in EXAMPLE_COUTRIES:
    predictions, test_y = prepare_country_data_for_plots(country_name, enc_names, rescaled_predictions)
    plot_pred_v_data(predictions, test_y, country_name)

plt.show()
