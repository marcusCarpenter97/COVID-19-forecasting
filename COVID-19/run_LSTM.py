import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import LSTM
import data_handler
import hyperparameter_search

forecast_horizon = 4   # Number of observations to be used to predict the next event.
train_set_ratio = 0.7  # The size of the training set as a percentage of the data set.

# Setup data.
raw_data = data_handler.load_data()
current_infected = data_handler.calculate_total_infected(raw_data)

# Split the data into train and test sets.
train, test = data_handler.split_train_test(current_infected, train_set_ratio)

# Make the time series data stationary.
train_log, train_diff, stationary_train = data_handler.adjust_data(train)
test_log, test_diff, stationary_test = data_handler.adjust_data(test)

# Transform the data into a supervised learning dataset.
supervised_train = data_handler.series_to_supervised(stationary_train, 0, forecast_horizon)
supervised_test = data_handler.series_to_supervised(stationary_test, 0, forecast_horizon)

# Create sets for input and output based on the forecast horizon.
x_train, y_train = data_handler.split_horizon(supervised_train, forecast_horizon)
x_test, y_test = data_handler.split_horizon(supervised_test, forecast_horizon)

# Reshape x from [samples, time steps] to [samples, time steps, features]
# Where samples = rows and time steps = columns.
features = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], features)

# Generate a list of hyperparameters through Random Search. TODO exception handling.
methods = {1 : "random", 2 : "taguchi"}
method = int(input("Select hyperparameter search method:\n1. Random\n2. Taguchi\n"))
if method == 1:
    hyper_sample_num = int(input("Enter number of hyperparameter combinations to use: "))
    hyper_samples = hyperparameter_search.select_hyperparameters(num_of_samples=hyper_sample_num, method=methods[method])
else:
    hyper_samples = hyperparameter_search.select_hyperparameters(method=methods[method])

# Create an LSTM model for each combination of hyperparameters.
lstms = []
for hyper_sample in hyper_samples:
    lstm = LSTM.myLSTM()
    lstm.hyper_params = hyper_sample
    lstm.create_simple_LSTM(nodes=hyper_sample[0],
                                dropout=hyper_sample[1],
                                loss=hyper_sample[2],
                                lstm_activation=hyper_sample[3],
                                dense_activation=hyper_sample[4])
    lstms.append(lstm)

# Train all of the created models.
epochs = int(input("Enter number of training epochs (must be > 0): "))
for idx, lstm in enumerate(lstms):
    print(f"Training model {idx} out of {len(lstms)}")
    lstm.train(x_train, y_train, epochs)
print("Done")

# Rescale answers to calculate the error.
scaled_train = data_handler.rescale_data(y_train, train_diff[0], train_log[0], forecast_horizon)
scaled_test = data_handler.rescale_data(y_test, test_diff[0], test_log[0], forecast_horizon)

# Generate performance reports for all models.
for idx, lstm in enumerate(lstms):
    model_name = str(idx)
    # Create predictions for the train and test data.
    train_prediction = lstm.predict(x_train)
    test_prediction = lstm.predict(x_test)

    # Rescale predictions.
    lstm.train_predictions = data_handler.rescale_data(train_prediction, train_diff[0], train_log[0], forecast_horizon)
    lstm.test_predictions = data_handler.rescale_data(test_prediction, test_diff[0], test_log[0], forecast_horizon)

    lstm.plot_history(model_name)
    lstm.plot_predictions(model_name, current_infected, forecast_horizon)

    print(f"\n\nModel name: {model_name}")
    print(f"RMSE on train: {lstm.rmse(scaled_train, lstm.train_predictions)}")
    print(f"RMSE on test: {lstm.rmse(scaled_test, lstm.test_predictions)}")

    print(f"RMSLE on train: {lstm.rmsle(scaled_train, lstm.train_predictions)}")
    print(f"RMSLE on test: {lstm.rmsle(scaled_test, lstm.test_predictions)}")

    print(f"Hyperparameters used: {lstm.hyper_params}")
