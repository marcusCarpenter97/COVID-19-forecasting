import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow messages.
import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import model

class Experiment():
    def __init__(self, val_scalers, test_scalers):
        self.logger = logging.getLogger("experiment")
        self.save_dir = "cross_val_results"
        self.ensemble_size = 10
        self.units = 20
        self.epochs = 300
        self.val_scalers = val_scalers
        self.test_scalers = test_scalers
        self.tanh = tf.keras.activations.tanh
        self.lstm = tf.keras.layers.LSTM
        self.gru = tf.keras.layers.GRU
        self.adam = tf.keras.optimizers.Adam()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]
        self.regularizers = [[0,    0,    0],    # No regularization
                             [0.01, 0,    0],    # Default L1 only.
                             [0,    0.01, 0],    # Default L2 only.
                             [0,    0,    0.2],  # Small dropout only.
                             [0.01, 0.01, 0],    # Default L1 and L2, no dropout.
                             [0.01, 0.01, 0.2]   # All regularizers.
                            ]

    def save_to_npz_folds(self, data, file_name):
        """
        Parameters:
            data - list containing numpy arrays with length = folds.
            file_name - string.
        """
        file_path = os.path.join(self.save_dir, file_name)
        with open(file_path, "w+b") as new_file:
            np.savez(new_file, *data)

    def destandardize_data(self, data, scalers):
        """
        data - numpy array of shape (countries, horizon, features)
        returns numpy array of shape shape as data
        """
        rescaled_data = []
        for country_pred, scaler in zip(data, scalers):
            rescaled_data.append(scaler.inverse_transform(country_pred))
        return np.stack(rescaled_data)

    def prepare_predictions(self, data, test_scalers):
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

        return self.destandardize_data(reshaped_preds, test_scalers)

    def calculate_rmse(self, orig, pred):
        """
        orig - numpy array of shape (countries, horizon, features)
        pred - numpy array of shape (countries, horizon, features)
        Returns numpy array of shape (countries, features) containing the RMSE of all predictions
        """
        results = []
        for o, p in zip(orig, pred):
            results.append(mean_squared_error(o, p, multioutput='raw_values', squared=False))
        return np.stack(results)

    def calculate_mae(self, orig, pred):
        """
        orig - numpy array of shape (countries, horizon, features)
        pred - numpy array of shape (countries, horizon, features)
        Returns numpy array of shape (countries, features) containing the MAE of all predictions
        """
        results = []
        for o, p in zip(orig, pred):
            results.append(mean_absolute_error(o, p, multioutput='raw_values'))
        return np.stack(results)

    def save_to_npz(self, data, file_name, exp_name):
        file_path = os.path.join(self.save_dir, f"{file_name}_{exp_name}")
        with open(file_path, "w+b") as new_file:
            np.savez(new_file, data)

    def run_experiments(self, horizon, padding, train, val, test_x, test_y, orig_val, orig_test_y, enc_names):
        fold_idx = 0
        data_zip = zip(train, val, test_x, test_y)
        for tr, v, te_x, te_y in data_zip:
            for reg_idx, regularizer in enumerate(self.regularizers):
                for trial in range(self.ensemble_size):
                    lstm = model.Model(horizon, self.units, self.lstm, self.tanh, regularizer[0], regularizer[1], regularizer[2],
                                       padding)
                    gru = model.Model(horizon, self.units, self.gru, self.tanh, regularizer[0], regularizer[1], regularizer[2],
                                      padding)

                    lstm.compile(self.adam, self.mse_loss, self.metrics)
                    gru.compile(self.adam, self.mse_loss, self.metrics)

                    self.logger.info(f"Training LSTM fold {fold_idx} regularizer {reg_idx} ensemble {trial}")
                    lstm_hist = lstm.fit([tr, enc_names], [v[0], v[1], v[2]], self.epochs, verbose=0)
                    self.logger.info(f"Training GRU fold {fold_idx} regularizer {reg_idx} ensemble {trial}")
                    gru_hist = gru.fit([tr, enc_names], [v[0], v[1], v[2]], self.epochs, verbose=0)

                    # make predictions on validation data
                    lstm_pred_val = lstm.predict([tr, enc_names])
                    gru_pred_val = gru.predict([tr, enc_names])

                    # make predictions on test data
                    lstm_pred_test = lstm.predict([te_x, enc_names])
                    gru_pred_test = gru.predict([te_x, enc_names])

                    # rescale validation predictions
                    lstm_pred_val = self.prepare_predictions(lstm_pred_val, self.val_scalers[fold_idx])
                    gru_pred_val = self.prepare_predictions(gru_pred_val, self.val_scalers[fold_idx])

                    # rescale test predictions
                    lstm_pred_test = self.prepare_predictions(lstm_pred_test, self.test_scalers[fold_idx])
                    gru_pred_test = self.prepare_predictions(gru_pred_test, self.test_scalers[fold_idx])

                    # calculate RMSE for validation
                    lstm_rmse_val = self.calculate_rmse(orig_val[fold_idx], lstm_pred_val)
                    gru_rmse_val = self.calculate_rmse(orig_val[fold_idx], gru_pred_val)

                    # calculate RMSE for test
                    lstm_rmse_test = self.calculate_rmse(orig_test_y[fold_idx], lstm_pred_test)
                    gru_rmse_test = self.calculate_rmse(orig_test_y[fold_idx], gru_pred_test)

                    # calculate MAE for validation 
                    lstm_mae_val = self.calculate_mae(orig_val[fold_idx], lstm_pred_val)
                    gru_mae_val = self.calculate_mae(orig_val[fold_idx], gru_pred_val)

                    # calculate MAE for test
                    lstm_mae_test = self.calculate_mae(orig_test_y[fold_idx], lstm_pred_test)
                    gru_mae_test = self.calculate_mae(orig_test_y[fold_idx], gru_pred_test)

                    lstm_name = f"fold_{fold_idx}_reg_{reg_idx}_ens_{trial}_lstm"
                    gru_name = f"fold_{fold_idx}_reg_{reg_idx}_ens_{trial}_gru"

                    # save model's predictions on validation data
                    self.save_to_npz(lstm_pred_val, lstm_name, "val_pred")
                    self.save_to_npz(gru_pred_val, gru_name, "val_pred")

                    # save model's predictions on test data
                    self.save_to_npz(lstm_pred_test, lstm_name, "test_pred")
                    self.save_to_npz(gru_pred_test, gru_name, "test_pred")

                    # save model's error scores for validation
                    self.save_to_npz(lstm_rmse_val, lstm_name, "rmse_val")
                    self.save_to_npz(gru_rmse_val, gru_name, "rmse_val")

                    self.save_to_npz(lstm_mae_val, lstm_name, "mae_val")
                    self.save_to_npz(gru_mae_val, gru_name, "mae_val")

                    # save model's error scores for test
                    self.save_to_npz(lstm_rmse_test, lstm_name, "rmse_test")
                    self.save_to_npz(gru_rmse_test, gru_name, "rmse_test")

                    self.save_to_npz(lstm_mae_test, lstm_name, "mae_test")
                    self.save_to_npz(gru_mae_test, gru_name, "mae_test")
            fold_idx += 1
