import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import LSTM
import data_handler

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

lstm = LSTM.myLSTM()
lstm.create_seq_simple_LSTM()
print("Training...")
lstm.train(x_train, y_train)
print("Done")

# Create predictions for the train and test data.
train_prediction = lstm.predict(x_train)
test_prediction = lstm.predict(x_test)

# Rescale predictions.
scaled_train_p = data_handler.rescale_data(train_prediction, train_diff[0], train_log[0], forecast_horizon)
scaled_test_p = data_handler.rescale_data(test_prediction, test_diff[0], test_log[0], forecast_horizon)
# Rescale answers.
scaled_train = data_handler.rescale_data(y_train, train_diff[0], train_log[0], forecast_horizon)
scaled_test = data_handler.rescale_data(y_test, test_diff[0], test_log[0], forecast_horizon)

train_rmse = lstm.rmse(scaled_train, scaled_train_p)
test_rmse = lstm.rmse(scaled_test, scaled_test_p)
print(f"Train RMSE : {train_rmse}")
print(f"Test RMSE : {test_rmse}")

# The first sample is lost after each differencing, so the + 2 is required. 
empty_arr = np.empty((forecast_horizon+2, 1))
empty_arr[:] = np.nan
# Join train and test predictions to create one curve.
predictions = np.concatenate([empty_arr, scaled_train_p, empty_arr, scaled_test_p])
# Plot it over the original data.

fig, ax = plt.subplots()
ax.plot(current_infected, label='Original data')
ax.plot(predictions, label='Forecasts')
legend = ax.legend(loc='best')
plt.xticks(rotation=90)
plt.show()
