#import operator
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

import LSTM
import data_handler
import hyperparameter_search

forecast_horizon = 4   # Number of observations to be used to predict the next event.
train_set_ratio = 0.7  # The size of the training set as a percentage of the data set.

# Setup data.
raw_data = data_handler.load_data()
current_infected = data_handler.calculate_current_infected(raw_data)

# Split the data into train and test sets.
train, test = data_handler.split_train_test(current_infected, train_set_ratio)

# Make the time series data stationary.
train_log, train_diff_one, train_diff_two, stationary_train = data_handler.adjust_data(train)
test_log, test_diff_one, test_diff_two, stationary_test = data_handler.adjust_data(test)

# Transform the data into a supervised learning dataset.
supervised_train = data_handler.series_to_supervised(stationary_train, 0, forecast_horizon)
supervised_test = data_handler.series_to_supervised(stationary_test, 0, forecast_horizon)

# Create sets for input and output based on the forecast horizon.
x_train, y_train = data_handler.split_horizon(supervised_train, forecast_horizon)
x_test, y_test = data_handler.split_horizon(supervised_test, forecast_horizon)

# Rescale answers to calculate the error.
scaled_train = data_handler.rescale_data(y_train, train_diff_one[0], train_diff_two[0], train_log[0], forecast_horizon)
scaled_test = data_handler.rescale_data(y_test, test_diff_one[0], test_diff_two[0], test_log[0], forecast_horizon)

# Reshape x from [samples, time steps] to [samples, time steps, features]
# Where samples = rows and time steps = columns.
features = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], features)

# Generate a list of hyperparameters through Random Search. TODO exception handling.
methods = {1 : "random", 2 : "taguchi", 3: "default"}
method = int(input("Select hyperparameter search method:\n1. Random\n2. Taguchi\n3. Default\n"))
if method == 1:
    hyper_sample_num = int(input("Enter number of hyperparameter combinations to use: "))
    hyper_samples = hyperparameter_search.select_hyperparameters(num_of_samples=hyper_sample_num, method=methods[method])
if method == 2:
    hyper_samples = hyperparameter_search.select_hyperparameters(method=methods[method])
else:  # Hand picked values.
    hyper_samples = [[5, 0.1, "mean_squared_error", "tanh", "sigmoid"]]

epochs = int(input("Enter number of training epochs (must be > 0): "))
n_tests = int(input("Enter number of tests (must be > 0): "))

# Create an LSTM model for each combination of hyperparameters.
for idx, hyper_sample in enumerate(hyper_samples):
    rmse_train = []
    rmse_test = []
    rmsle_train = []
    rmsle_test = []
    mase_train = []
    mase_test = []

    for _ in range(n_tests):
        lstm = LSTM.myLSTM()
        lstm.hyper_params = hyper_sample
        lstm.create_simple_LSTM(nodes=hyper_sample[0],
                                dropout=hyper_sample[1],
                                loss=hyper_sample[2],
                                lstm_activation=hyper_sample[3],
                                dense_activation=hyper_sample[4])

        print(f"Training model with hyperparameters: {hyper_sample}")
        lstm.train(x_train, y_train, epochs)
        print("Done")

        # Create predictions for the train and test data.
        train_prediction = lstm.predict(x_train)
        test_prediction = lstm.predict(x_test)

        # Rescale predictions.
        lstm.train_predictions = data_handler.rescale_data(train_prediction, train_diff_one[0],
                                                           train_diff_two[0], train_log[0],
                                                           forecast_horizon)
        lstm.test_predictions = data_handler.rescale_data(test_prediction, test_diff_one[0],
                                                          test_diff_two[0], test_log[0],
                                                          forecast_horizon)

        lstm.plot_history(idx)
        lstm.plot_predictions(idx, current_infected, forecast_horizon)

        rmse_train.append(lstm.rmse(lstm.train_predictions, scaled_train))
        rmse_test.append(lstm.rmse(lstm.test_predictions, scaled_test))

        rmsle_train.append(lstm.rmsle(lstm.train_predictions, scaled_train))
        rmsle_test.append(lstm.rmsle(lstm.test_predictions, scaled_test))

        mase_train.append(lstm.mase(lstm.train_predictions, scaled_train, len(x_train)))
        mase_test.append(lstm.mase(lstm.test_predictions, scaled_test, len(x_train)))

    # Generate performance report for model.
    report = f"\n\nModel name: {idx}, Number of tests: {n_tests}, Number of epochs: {epochs}\n RMSE on train: {rmse_train}\n RMSE on test: {rmse_test}\n Average RMSE: {np.array(rmse_test).mean()}\n RMSLE on train: {rmsle_train}\n RMSLE on test: {rmsle_test}\n Average RMSLE: {np.array(rmsle_test).mean()}\n MASE on train: {mase_train}\n MASE on test: {mase_test}\n Average MASE: {np.array(mase_test).mean()}\n Hyperparameters used: {lstm.hyper_params}\n"

    with open("LSTM_results.txt", 'a') as res_file:
        res_file.write(report)
