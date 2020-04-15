import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import LSTM
import data_handler

forecast_horizon = 4   # Number of observations to be used to predict the next event.
train_set_size = 0.67  # The size of the training set as a percentage of the data set.

#Load in the data most recent data available on GitHub.
raw_data = data_handler.load_data()

# Calculate the current total cases from the data.
current_infected = data_handler.calculate_total_infected(raw_data)

# First the natural logarithm of the original data is taken.
log_cases = pd.DataFrame(np.log(current_infected))

# Then the difference between each point in the data is calculated.
first_diff = log_cases.diff().dropna()

# As the first differencing did not provide good results the difference of the difference is taken.
stationary_data = first_diff.diff().dropna()

# Convert the time series to a supervised learning dataset.
# In this setting four days (input) of data will be used to forecast the fith day (output).
supervised_data = data_handler.series_to_supervised(stationary_data, 0, forecast_horizon)

# Split the data into input (x) and output (y).
x, y = data_handler.split_data(supervised_data, forecast_horizon)

# Split the data into train and test sets.
train_size = int(x.shape[0] * train_set_size)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

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

#diff_list = list(first_diff[0])
#log_list = list(log_cases[0])

predictions = np.concatenate((train_prediction, test_prediction))
#scaled_predictions = data_handler.rescale_data(predictions, diff_list, log_list, forecast_horizon)
scaled_predictions = data_handler.rescale_data(predictions, first_diff[0], log_cases[0], forecast_horizon)

#scaled_y = data_handler.rescale_data(y, diff_list, log_list, forecast_horizon)
scaled_y = data_handler.rescale_data(y, first_diff[0], log_cases[0], forecast_horizon)

train_rmse = lstm.rmse(y_train, train_prediction)
test_rmse = lstm.rmse(y_test, test_prediction)
total_rmse = lstm.rmse(scaled_y, scaled_predictions).to_numpy()  # Because the inputs are DataFrames.
print(f"Train RMSE : {train_rmse}")
print(f"Test RMSE : {test_rmse}")
print(f"Total RMSE : {total_rmse[0]}")

fig, ax = plt.subplots()
ax.plot(scaled_y, label='Original data')
ax.plot(scaled_predictions, label='Forecasts')
legend = ax.legend(loc='best')
plt.show()
