import os
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import data
import models
from utils import reshape_predictions

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
    data - list
    """
    rescaled_data = []
    for fold, scaler_fold in zip(data, scalers):
        rescaled_fold = []
        for country, scaler in zip(fold, scaler_fold):
            rescaled_fold.append(scaler.inverse_transform(country))
        rescaled_data.append(np.stack(rescaled_fold))
    return rescaled_data

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

def prepare_predictions(data):
    """
    data - list
    """
    data = np.stack(data)
    data = reshape_predictions(data)
    return D.destandarize_data(data)

def pickle_data(data, file_name):
    file_path = os.path.join(SAVE_DIR, file_name)
    with open(file_path, "wb") as new_file:
        pickle.dump(data, new_file, protocol=pickle.DEFAULT_PROTOCOL)

def save_to_csv(data, file_name):
    file_path = os.path.join(SAVE_DIR, file_name)
    temp_df = pd.DataFrame(data)
    temp_df.to_csv(file_path, index=False)

def save_to_npz(data, file_name):
    file_path = os.path.join(SAVE_DIR, file_name)
    with open(file_path, "wb") as new_file:
        np.savez(new_file, data)

if __name__ == "__main__":

    D = data.Data()  # This loads the data.

    enc_names = D.encode_names()  # Encode the countrie's names.

    # Prepare all the k-foldas fot the cross validation.
    train, val, test_x, test_y = split_data(D, H)

    idx = 1
    for tr, v, te_x, te_y in zip(train, val, test_x, test_y):
        print(f"{idx}")
        idx += 1
        print(f"train fold {tr.shape}")
        print(f"val fold {v.shape}")
        print(f"test input fold {te_x.shape}")
        print(f"test output fold {te_y.shape}")

    print("Pre scale example")
    print(test_y[2][0])

    # Standardize all the data to make it easier to train the model.
    scaled_train, _ = standardize_data(train)
    scaled_val, _ = standardize_data(val)
    scaled_test_x, _ = standardize_data(test_x)
    scaled_test_y, test_y_scalers = standardize_data(test_y)

    print(scaled_test_y[2][0])

    rescaled_test_y = destandardize_data(scaled_test_y, test_y_scalers)
    print(rescaled_test_y[2][0])

    # Pad the input data as they all must be in the same shape.
    # The cross validation makes each sample a different size.
    padded_scaled_train = pad_data(scaled_train)
    padded_scaled_test_x = pad_data(scaled_test_x, offset=1)

    print(f"padded scaled train : {len(padded_scaled_train)}")
    for fold in padded_scaled_train:
        print(f"padded scaled train : {fold.shape}")

    print(f"padded scaled test x: {len(padded_scaled_test_x)}")
    for fold in padded_scaled_test_x:
        print(f"padded scaled test x: {fold.shape}")

    # The output data must be split by features as the model used a separate branch for each of them.
    multi_out_scaled_val = prepare_output_data(scaled_val)
    multi_out_scaled_test_y = prepare_output_data(scaled_test_y)
    print(f"Scaled val len: {len(scaled_val)}")
    print(f"Scaled val shape: {scaled_val[0].shape}")
    print(f"Multi out scaled val len: {len(multi_out_scaled_val)}")
    print(f"Multi out scaled val shape: {multi_out_scaled_val[0].shape}")

    print(f"Scaled test_y len: {len(scaled_test_y)}")
    print(f"Scaled test_y shape: {scaled_test_y[0].shape}")
    print(f"Multi out scaled test_y len: {len(multi_out_scaled_test_y)}")
    print(f"Multi out scaled test_y shape: {multi_out_scaled_test_y[0].shape}")

    # Validation loop.
    fold_idx = 1
    data = zip(padded_scaled_train, padded_scaled_test_x, multi_out_scaled_val, multi_out_scaled_test_y)
    for tr, te_x, v, te_y in data:
        for reg_idx, reg_val in enumerate(REG_VALS):
            # create models
            lstm_model = models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["lstm"], ACTIVATIONS["tanh"], PAD_VAL,
                                                         l1=reg_val[0], l2=reg_val[1], dropout=reg_val[2])
            gru_model = models.RNNMultiOutputIndividual(OUTPUT_SIZE, UNITS, RNN_LAYERS["gru"], ACTIVATIONS["tanh"], PAD_VAL,
                                                        l1=reg_val[0], l2=reg_val[1], dropout=reg_val[2])

            # compile models
            models.compileModel(lstm_model, COMPILE_PARAMS["optimizer"], COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])
            models.compileModel(gru_model, COMPILE_PARAMS["optimizer"], COMPILE_PARAMS["loss"], COMPILE_PARAMS["metrics"])

            # train models
            lstm_hist = models.fitModel(lstm_model, [tr, enc_names], [v[0], v[1], v[2]], EPOCHS, verbose=0)
            gru_hist = models.fitModel(gru_model, [tr, enc_names], [v[0], v[1], v[2]], EPOCHS, verbose=0)

            # evaluate models
            lstm_eval = models.evaluateModel(lstm_model, x=[te_x, enc_names], y=[te_y[0], te_y[1], te_y[2]])
            gru_eval = models.evaluateModel(gru_model, x=[te_x, enc_names], y=[te_y[0], te_y[1], te_y[2]])

            # make predictions
            lstm_pred = lstm_model.predict([te_x, enc_names])
            gru_pred = gru_model.predict([te_x, enc_names])

            # rescale predictions
            lstm_pred = prepare_predictions(lstm_pred)
            gru_pred = prepare_predictions(gru_pred)

            # calculate RMSE
            lstm_errors = D.calculate_error(lstm_pred)
            gru_errors = D.calculate_error(gru_pred)

            # make file names for saving results
            lstm_name = f"lstm_reg{reg_idx}_fold{fold_idx}"
            gru_name = f"gru_reg{reg_idx}_fold{fold_idx}"

            # save model's training history
            pickle_data(lstm_hist, lstm_name)
            pickle_data(gru_hist, gru_name)

            # save model's evaluation results
            save_to_csv(lstm_eval, lstm_name)
            save_to_csv(gru_eval, gru_name)

            # save model's predictions
            save_to_npz(lstm_pred, lstm_name)
            save_to_npz(gru_pred, gru_name)

            # save model's error scores
            save_to_csv(lstm_errors, lstm_name)
            save_to_csv(gru_errors, gru_name)

            fold_idx += 1

    # save original data
    orig_data = {"train": padded_scaled_train,
                 "test_x": padded_scaled_test_x,
                 "validation": multi_out_scaled_val,
                 "test_y": multi_out_scaled_test_y}

    file_path = os.path.join(SAVE_DIR, "original_data")
    orig_df = pd.DataFrame(orig_data)
    orig_df.to_csv(file_path, index=False)
