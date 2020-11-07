import models
from datastructures import *
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import *

# Get command line options
parser = argparse.ArgumentParser()
parser.add_argument("--load_all", action="store_true", help="Load saved models.")
args = parser.parse_args()

REG_VALS = [[0, 0, 0],  # No regularization
            [0.01, 0, 0],  # Default L1 only.
            [0, 0.01, 0],  # Default L2 only.
            [0, 0, 0.2],  # Small dropout only.
            [0.01, 0.01, 0],  # Default L1 and L2, no dropout.
            [0.01, 0.01, 0.2]  # All regularizers.
            ]

GRU_CHECKPOINTS = ["checkpoints/gru_no_reg.ckpt",
                   "checkpoints/gru_l1.ckpt",
                   "checkpoints/gru_l2.ckpt",
                   "checkpoints/gru_drop.ckpt",
                   "checkpoints/gru_l1l2.ckpt",
                   "checkpoints/gru_all.ckpt"]

LSTM_CHECKPOINTS = ["checkpoints/lstm_no_reg.ckpt",
                    "checkpoints/lstm_l1.ckpt",
                    "checkpoints/lstm_l2.ckpt",
                    "checkpoints/lstm_drop.ckpt",
                    "checkpoints/lstm_l1l2.ckpt",
                    "checkpoints/lstm_all.ckpt"]

# Create models.
lstm_experiments = []
for reg_val in REG_VALS:
    lstm_experiments.append(models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"], ACTIVATIONS["tanh"],
                                                            l1=reg_val[0], l2=reg_val[1], dropout=reg_val[2]))

gru_experiments = []
for reg_val in REG_VALS:
    gru_experiments.append(models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"], ACTIVATIONS["tanh"],
                                                           l1=reg_val[0], l2=reg_val[1], dropout=reg_val[2]))

# Compile models.
for gru in gru_experiments:
    models.compileModel(gru, COMPILE_PARAMS["optimizer"], COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])

for lstm in lstm_experiments:
    models.compileModel(lstm, COMPILE_PARAMS["optimizer"], COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])

if args.load_all:
    for idx, gru in enumerate(gru_experiments):
        gru.load_weights(GRU_CHECKPOINTS[idx])
    for idx, lstm in enumerate(lstm_experiments):
        lstm.load_weights(LSTM_CHECKPOINTS[idx])
else:
    # Train models.
    gru_hist = []
    for idx, gru in enumerate(gru_experiments):
        print(f"Training GRU {idx}/{len(gru_experiments)}")
        checkpoint = ModelCheckpoint(filepath=GRU_CHECKPOINTS[idx], save_weights_only=True)
        gru_hist.append(models.fitModel(gru, INPUT_TRAIN_DATA, OUTPUT_TRAIN_DATA["multiOutput"], EPOCHS, [checkpoint], verbose=0))

    lstm_hist = []
    for idx, lstm in enumerate(lstm_experiments):
        print(f"Training LSTM {idx}/{len(lstm_experiments)}")
        checkpoint = ModelCheckpoint(filepath=LSTM_CHECKPOINTS[idx], save_weights_only=True)
        lstm_hist.append(models.fitModel(lstm, INPUT_TRAIN_DATA, OUTPUT_TRAIN_DATA["multiOutput"], EPOCHS, [checkpoint],
                                         verbose=0))
# Evaluate models.
gru_eval = []
for idx, gru in enumerate(gru_experiments):
    gru_eval.append(models.evaluateModel(gru, x=INPUT_TEST_DATA, y=OUTPUT_TEST_DATA["multiOutput"]))

lstm_eval = []
for idx, lstm in enumerate(lstm_experiments):
    lstm_eval.append(models.evaluateModel(lstm, x=INPUT_TEST_DATA, y=OUTPUT_TEST_DATA["multiOutput"]))

# Make predictions.
gru_predictions = []
for gru in gru_experiments:
    pred = gru.predict(INPUT_TEST_DATA)
    pred = np.stack(pred)
    pred = reshape_predictions(pred)
    pred = COVID_DATA.destandarize_data(pred)
    gru_predictions.append(pred)

lstm_predictions = []
for lstm in lstm_experiments:
    pred = lstm.predict(INPUT_TEST_DATA)
    pred = np.stack(pred)
    pred = reshape_predictions(pred)
    pred = COVID_DATA.destandarize_data(pred)
    lstm_predictions.append(pred)

# Calculate MSE and RMSE for all forecasts for each country.
gru_errors = []
for gru_pred in gru_predictions:
    gru_errors.append(COVID_DATA.calculate_error(gru_pred))

lstm_errors = []
for lstm_pred in lstm_predictions:
    lstm_errors.append(COVID_DATA.calculate_error(lstm_pred))

# Print errors for each model for the sample countries.
for idx, gru_error in enumerate(gru_errors):
    print(f"GRU model {idx}")
    print_error_scores(gru_error, EXAMPLE_COUTRIES)

for idx, lstm_error in enumerate(lstm_errors):
    print(f"LSTM model {idx}")
    print_error_scores(lstm_error, EXAMPLE_COUTRIES)

# Make plot with orig data and all forecasts for UK.
country = COVID_DATA.find_country("United Kingdom")
country_idx = (enc_names == country.encoded_name).all(axis=1).nonzero()

# Get UK predictions from the results.
gru_preds_to_plot = []
for gru_pred in gru_predictions:
    country_predictions = gru_pred[country_idx]
    gru_preds_to_plot.append(country_predictions.reshape(country_predictions.shape[1], country_predictions.shape[2]).T)

lstm_preds_to_plot = []
for lstm_pred in lstm_predictions:
    country_predictions = lstm_pred[country_idx]
    lstm_preds_to_plot.append(country_predictions.reshape(country_predictions.shape[1], country_predictions.shape[2]).T)

# Plot all the models with the original data.
def plot_reg_res(model_type, preds_to_plot):
    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle(f"Effects of regularizing the {model_type}")
    for feature_idx, ax in enumerate(axes):
        ax.plot(country.test_y.T[feature_idx])
        for pred in preds_to_plot:
            ax.plot(pred[feature_idx])
        ax.set_title(sub_titles[feature_idx])
        ax.set_xlabel("Days")
        ax.set_ylabel("People")
        ax.legend(["Original", "No reg", "L1 = 0.01", "L2 = 0.01", "dropout = 0.2", "L1_L2 = 0.01", "All reg"])

plot_reg_res("GRU", gru_preds_to_plot)
plot_reg_res("LSTM", lstm_preds_to_plot)

def result_generator():
    for country in EXAMPLE_COUTRIES:
        gru_res, lstm_res = [], []
        for gru_experiment, lstm_experiment in zip(gru_errors, lstm_errors):
            gru_res.append(gru_experiment[country]["RMSE"])
            lstm_res.append(lstm_experiment[country]["RMSE"])
        yield np.stack(gru_res).T, np.stack(lstm_res).T

res_gen = result_generator()

exp_num = 6  # six experiments
bar_idx = np.arange(exp_num)  # Positions for bar on plot.
width = 0.3  # width for bars in plot.

# Plot histograms of RMSE comparing LSTM and GRU.
sub_titles = ["Confirmed", "Deceased", "Recovered"]
fig, axes = plt.subplots(len(EXAMPLE_COUTRIES), 3, constrained_layout=True)
#fig.suptitle("Comparison of RMSE for LSTM (blue) and GRU (orange) models with different regularization methods")
for row_idx, row in enumerate(axes):
    gru_res, lstm_res = next(res_gen)
    for column_idx, column in enumerate(row):
        column.bar(bar_idx, gru_res[column_idx], width, color="orange")
        column.bar(bar_idx+width, lstm_res[column_idx], width, color="b")
        column.set_xticks([0, 1, 2, 3, 4, 5])
        column.set_xticklabels(["No reg", "L1", "L2", "Dropout", "L1L2", "All reg"])
        # Only add title for columns on the first row
        if row_idx == 0:
            column.set_title(sub_titles[column_idx])
        # Add the country's name to the y axis on the first column
        if column_idx == 0:
            if EXAMPLE_COUTRIES[row_idx] == "United Kingdom":
                column.set_ylabel("UK")
            else:
                column.set_ylabel(EXAMPLE_COUTRIES[row_idx])
fig.align_ylabels()

plt.show()
