import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import models
from datastructures import *
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduces tensorflow messages.

# Get command line options
parser = argparse.ArgumentParser()
parser.add_argument("--load_all", action="store_true", help="Load saved models.")
parser.add_argument("--train_specific", action="store_true", help="Only train certain models for targeted testing.")
args = parser.parse_args()

compile_models(MULTI_MODELS, COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])
compile_models(SINGLE_MODELS, COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])
compile_models(MULTI_QUANTILE_MODELS, MULTI_QUANTILE_LOSSES, QUANTILE_METRICS)
compile_models(SINGLE_QUANTILE_MODELS, QUANTILE_LOSSES, QUANTILE_METRICS)

if args.train_specific:
    print("Training specific models...")
    # Select sub set of dict from main dict.
    models = {k:MULTI_MODELS[k] for k in ("multiOutIndvLstm","multiOutIndvGru") if k in MULTI_MODELS}
    multi_out_hist = train_models(models, "multiOutput")
    print("Done.")
    multi_out_eval = evaluate_models(models, "multiOutput")
    print_eval_res(multi_out_eval)
else:
    if args.load_all:
        print("Loading models...")
        load_models(MULTI_MODELS)
        load_models(SINGLE_MODELS)
        load_models(MULTI_QUANTILE_MODELS)
        load_models(SINGLE_QUANTILE_MODELS)
        print("Done.")
    else:
        print("Training all models...")
        multi_out_hist = train_models(MULTI_MODELS, "multiOutput")
        single_out_hist = train_models(SINGLE_MODELS, "singleOutput")
        multi_quantile_out_hist = train_models(MULTI_QUANTILE_MODELS, "multiOutQuant")
        single_quantile_out_hist = train_models(SINGLE_QUANTILE_MODELS, "singleOutQuant")
        print("Done.")

    # Evaluate models.
    multi_out_eval = evaluate_models(MULTI_MODELS, "multiOutput")
    single_out_eval = evaluate_models(SINGLE_MODELS, "singleOutput")
    multi_quantile_out_eval = evaluate_models(MULTI_QUANTILE_MODELS, "multiOutQuant")
    single_quantile_out_eval = evaluate_models(SINGLE_QUANTILE_MODELS, "singleOutQuant")

    # Print evalutaion results.
    print_eval_res(multi_out_eval)
    print_eval_res(single_out_eval)
    print_eval_res(multi_quantile_out_eval)
    print_eval_res(single_quantile_out_eval)

# Predictions using best model on a selection of countries.
mulit_out_pred = MULTI_MODELS["multiOutIndvGru"].predict([test_x, enc_names])
mulit_out_pred = np.stack(mulit_out_pred)

# Reshape the predictions to fit the rescaler.
predictions = []
for c, d, r in zip(mulit_out_pred[0], mulit_out_pred[1], mulit_out_pred[2]):
    predictions.append(np.stack([c, d, r]).T)

predictions = np.stack(predictions)

# De-standarize predictions and data.
rescaled_predictions = COVID_DATA.destandarize_data(predictions)

# Plot predictions vs data.
def prepare_country_data_for_plots(name, enc_names, prediction_data):
    country = COVID_DATA.find_country(name)
    country_idx = (enc_names == country.encoded_name).all(axis=1).nonzero()
    country_predictions = prediction_data[country_idx]
    country_predictions = country_predictions.reshape(country_predictions.shape[1], country_predictions.shape[2]).T
    return country_predictions, country.test_y.T

for country_name in EXAMPLE_COUTRIES:
    predictions, test_y = prepare_country_data_for_plots(country_name, enc_names, rescaled_predictions)
    plot_pred_v_data(predictions, test_y, country_name)

plt.show()
