import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Rescale data.
scaled_train = data_handler.rescale_data(y_train, train_diff[0], train_log[0], forecast_horizon)
scaled_test = data_handler.rescale_data(y_test, test_diff[0], test_log[0], forecast_horizon)

# Plot data.
#cur_inf_fig = data_handler.plot(current_infected)
#train_fig = data_handler.plot(train)
#test_fig = data_handler.plot(test)
#stat_train_fig = data_handler.plot(stationary_train)
#stat_test_fig = data_handler.plot(stationary_test)

#strain_fig = data_handler.plot(scaled_train)
#stest_fig = data_handler.plot(scaled_test)

# The first sample is lost after each differencing, so the + 2 is required. 
empty_arr = np.empty((forecast_horizon+2, 1))
empty_arr[:] = np.nan
og = np.concatenate([empty_arr, scaled_train, empty_arr, scaled_test])

fig = plt.figure() 
plt.plot(current_infected)
plt.plot(og)

plt.show()
