""" Utility module woth some helper functions. """
import os
import numpy as np
from pprint import pprint
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from datastructures import *
import models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduces tensorflow messages.

def compile_models(model_dict: dict, losses: list, metrics: list):
    """
    params: model_dict - contains models to be compiled.
            losses - loss functions for the models.
            metrics - error and performance etrics for the models.
    returns: None
    """
    for name, model in model_dict.items():
        print(f"Compiling model: {name}")
        models.compileModel(model, COMPILE_PARAMS["optimizer"], losses, metrics)

def train_models(model_dict: dict, out_data: str) -> dict:
    """
    params: model_dict - contains models to be compiled.
            out_data - key for selecting which output data to use.
    returns: models_hist - contains the training history for all the models.
    """
    models_hist = {}
    for name, model in model_dict.items():
        print(f"Training model: {name}")
        checkpoint = ModelCheckpoint(filepath=CHECKPOINT_PATHS[name], save_weights_only=True)
        models_hist[name] = models.fitModel(model, x=INPUT_TRAIN_DATA, y=OUTPUT_TRAIN_DATA[out_data], epochs=EPOCHS,
                                            callbacks=[checkpoint])
    return models_hist

def load_models(models):
    """
    params:
    returns:
    """
    for name, model in models.items():
        model.load_weights(CHECKPOINT_PATHS[name])

def evaluate_models(model_dict, out_data):
    """
    params:
    returns:
    """
    models_eval = {}
    for name, model in model_dict.items():
        print(f"Evaluating model: {name}")
        models_eval[name] = models.evaluateModel(model, x=INPUT_TEST_DATA, y=OUTPUT_TEST_DATA[out_data])
    return models_eval

def print_eval_res(eval_dict):
    """
    params:
    returns:
    """
    for model_res in eval_dict:
        print(f"Evaluation results for {model_res}")
        pprint(eval_dict[model_res])
        print()

def plot_pred_v_data(pred, og_data, country_name, model_name):
    """
    params:
    returns:
    """
    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle(f"{country_name} - {model_name}")
    for idx, ax in enumerate(axes):
        ax.plot(pred[idx])
        ax.plot(og_data[idx])
        ax.set_title(sub_titles[idx])
        ax.set_xlabel("Days")
        ax.set_ylabel("People")
        ax.legend(["Model predictions", "Real data"], loc="best")

def reshape_predictions(mulit_out_pred):
    predictions = []
    for c, d, r in zip(mulit_out_pred[0], mulit_out_pred[1], mulit_out_pred[2]):
        predictions.append(np.stack([c, d, r]).T)
    return np.stack(predictions)
