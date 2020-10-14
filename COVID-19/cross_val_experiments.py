import os
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import data
import models

# Global variables
SAVE_DIR = "cross_val_results"
PAD_VAL = -10000
H = 28
F = 3
UNITS = 100
OUTPUT_SIZE = 28
EPOCHS = 300

ACTIVATIONS = {"tanh" : tf.keras.activations.tanh,
               "relu" : tf.keras.activations.relu}

RNN_LAYERS = {"lstm" : tf.keras.layers.LSTM,
              "gru" : tf.keras.layers.GRU}

COMPILE_PARAMS = {"optimizer" : tf.keras.optimizers.Adam(),
                  "loss" : tf.keras.losses.MeanSquaredError(),
                  "metrics" : [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]}

REG_VALS = [[0, 0, 0],  # No regularization
            [0.01, 0, 0],  # Default L1 only.
            [0, 0.01, 0],  # Default L2 only.
            [0, 0, 0.2],  # Small dropout only.
            [0.01, 0.01, 0],  # Default L1 and L2, no dropout.
            [0.01, 0.01, 0.2]  # All regularizers.
            ]

PADDING = np.full((H, F), PAD_VAL)

def split_data(data, horizon):
    train, val, test_x, test_y = [], [], [], []
    train_size = horizon
    # This assumes all countries have the same length.
    # The minus two gives space for the validation and test sets as they will overshoot.
    k_folds = len(data.countries[0].data)//horizon - 2
    for _ in range(k_folds):
        tr, v, te_x, te_y = data.cross_validate(train_size, horizon)
        train.append(tr), val.append(v), test_x.append(te_x), test_y.append(te_y)
        train_size += horizon
    return train, val, test_x, test_y

def standardize_data(data):
    """
    data - list
    """
    scaled_data = []
    scalers = []
    for fold in data:
        scaled_fold = []
        scalers_fold = []
        for country in fold:
            scaler = StandardScaler()
            scaled_fold.append(scaler.fit_transform(country))
            scalers_fold.append(scaler)
        scaled_data.append(np.stack(scaled_fold))
        scalers.append(scalers_fold)
    return scaled_data, scalers

def destandardize_data(data, scalers):
    """
    data - numpy array of shape (countries, horizon, features)
    returns numpy array of shape shape as data
    """
    rescaled_data = []
    for country_pred, scaler in zip(data, scalers):
        rescaled_data.append(scaler.inverse_transform(country_pred))
    return np.stack(rescaled_data)

def pad_data(data, offset=0):
    """
    data - list
    """
    padded_scaled_data = []
    folds = len(data) - offset
    for fold in data:
        padded_fold = []
        fold_padding = np.repeat(PADDING, folds).reshape(H*folds, F)
        for row in fold:
            padded_fold.append(np.append(row, fold_padding, axis=0))
        folds -= 1
        padded_scaled_data.append(np.stack(padded_fold))
    return padded_scaled_data

def prepare_output_data(data):
    """
    data - list
    """
    multi_out_data = []
    for fold in data:
        multi_out_data.append(D.make_multi_output_data(fold))
    return multi_out_data

def prepare_predictions(data, test_y_scalers):
    """
    data - list - contains one numpy array of shape (countries, horizon) for each feature.
    returns numpy array of shape (countries, horizon, features) with rescaled data.
    """
    # First reshape the data.
    reshaped_preds = []
    # Unpack all sub lists into the three variables.
    for c, d, r in zip(*data):
        reshaped_preds.append(np.stack([c, d, r]).T)
    reshaped_preds = np.stack(reshaped_preds)

    return destandardize_data(reshaped_preds, test_y_scalers)

def calculate_error(orig, pred):
    """
    orig - numpy array of shape (countries, horizon, features)
    pred - numpy array of shape (countries, horizon, features)
    Returns numpy array of shape (countries, features) containing the RMSE of all predictions
    """
    results = []
    for o, p in zip(orig, pred):
        results.append(mean_squared_error(o, p, multioutput='raw_values', squared=False))
    return np.stack(results)

def pickle_data(data, file_name, exp_name):
    file_path = os.path.join(SAVE_DIR, file_name%exp_name)
    with open(file_path, "wb") as new_file:
        pickle.dump(data, new_file, protocol=pickle.DEFAULT_PROTOCOL)

def save_to_csv(data, file_name, exp_name):
    file_path = os.path.join(SAVE_DIR, file_name%exp_name)
    temp_df = pd.DataFrame(data)
    temp_df.to_csv(file_path, index=False)

def save_to_npz(data, file_name, exp_name):
    file_path = os.path.join(SAVE_DIR, file_name%exp_name)
    with open(file_path, "wb") as new_file:
        np.savez(new_file, data)

if __name__ == "__main__":

    D = data.Data()  # This loads the data.

    enc_names = D.encode_names()  # Encode the countrie's names.

    # Prepare all the k-foldas fot the cross validation.
    train, val, test_x, test_y = split_data(D, H)

    # Standardize all the data to make it easier to train the model.
    scaled_train, _ = standardize_data(train)
    scaled_val, _ = standardize_data(val)
    scaled_test_x, _ = standardize_data(test_x)
    scaled_test_y, test_y_scalers = standardize_data(test_y)

    # Pad the input data as they all must be in the same shape.
    # The cross validation makes each sample a different size.
    padded_scaled_train = pad_data(scaled_train)
    padded_scaled_test_x = pad_data(scaled_test_x, offset=1)

    # The output data must be split by features as the model used a separate branch for each of them.
    multi_out_scaled_val = prepare_output_data(scaled_val)
    multi_out_scaled_test_y = prepare_output_data(scaled_test_y)

    # Validation loop.
    fold_idx = 1
    data = zip(padded_scaled_train, padded_scaled_test_x, multi_out_scaled_val, multi_out_scaled_test_y)
    for tr, te_x, v, te_y in data:
        print(f"Validation loop {fold_idx}")
        for reg_idx, reg_val in enumerate(REG_VALS):
            print(f"Regularizer loop {reg_idx}")
            # create models
            lstm_model = models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"], ACTIVATIONS["tanh"],
                                                         l1=reg_val[0], l2=reg_val[1], dropout=reg_val[2], pad_val=PAD_VAL)
            gru_model = models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"], ACTIVATIONS["tanh"],
                                                        l1=reg_val[0], l2=reg_val[1], dropout=reg_val[2], pad_val=PAD_VAL)

            # compile models
            models.compileModel(lstm_model, COMPILE_PARAMS["optimizer"], COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])
            models.compileModel(gru_model, COMPILE_PARAMS["optimizer"], COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])

            # train models
            print("Training LSTM")
            lstm_hist = models.fitModel(lstm_model, [tr, enc_names], [v[0], v[1], v[2]], EPOCHS, verbose=0)
            print("Training GRU")
            gru_hist = models.fitModel(gru_model, [tr, enc_names], [v[0], v[1], v[2]], EPOCHS, verbose=0)

            # evaluate models
            lstm_eval = models.evaluateModel(lstm_model, x=[te_x, enc_names], y=[te_y[0], te_y[1], te_y[2]], verbose=0)
            gru_eval = models.evaluateModel(gru_model, x=[te_x, enc_names], y=[te_y[0], te_y[1], te_y[2]], verbose=0)

            # make predictions
            lstm_pred = lstm_model.predict([te_x, enc_names])
            gru_pred = gru_model.predict([te_x, enc_names])

            # rescale predictions
            lstm_pred = prepare_predictions(lstm_pred, test_y_scalers[fold_idx])
            gru_pred = prepare_predictions(gru_pred, test_y_scalers[fold_idx])

            # calculate RMSE
            lstm_errors = calculate_error(scaled_test_y[fold_idx], lstm_pred)
            gru_errors = calculate_error(scaled_test_y[fold_idx], gru_pred)

            # make file names for saving results
            lstm_name = f"lstm_reg{reg_idx}_fold{fold_idx}_%s"
            gru_name = f"gru_reg{reg_idx}_fold{fold_idx}_%s"

            # save model's training history
            pickle_data(lstm_hist, lstm_name, "hist")
            pickle_data(gru_hist, gru_name, "hist")

            # save model's evaluation results
            save_to_csv(lstm_eval, lstm_name, "eval")
            save_to_csv(gru_eval, gru_name, "eval")

            # save model's predictions
            save_to_npz(lstm_pred, lstm_name, "pred")
            save_to_npz(gru_pred, gru_name, "pred")

            # save model's error scores
            save_to_npz(lstm_errors, lstm_name, "error")
            save_to_npz(gru_errors, gru_name, "errors")

            fold_idx += 1

    # save original data
    orig_data = {"train": padded_scaled_train,
                 "test_x": padded_scaled_test_x,
                 "validation": multi_out_scaled_val,
                 "test_y": multi_out_scaled_test_y}

    file_path = os.path.join(SAVE_DIR, "original_data")
    orig_df = pd.DataFrame(orig_data)
    orig_df.to_csv(file_path, index=False)
