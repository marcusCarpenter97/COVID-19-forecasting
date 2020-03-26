import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import LSTM, Dense
import data_handler

def series_to_supervised(data, before, after):
    new_cols = []
	
    for col in range(before, 0, -1):
        new_cols.append(data.shift(periods=col).rename(columns={0: -col}))

    new_cols.append(data)

    for col in range(1, after+1):
        new_cols.append(data.shift(periods=-col).rename(columns={0: col}))

    return pd.concat(new_cols, axis=1)

# prediction at time t = first original data + sum of all differenced predictions up to t.
def reverse_difference(original, diff_predictions):
    return [original + sum(diff_predictions[:t]) for t, _ in enumerate(diff_predictions)] # un-diff predictions

# Input: numpy arrays
def rmse(target, prediction):
    return np.sqrt(((target - prediction) ** 2).mean())

if __name__ == "__main__":
    #Load in the data most recent data available on GitHub.
    raw_data = data_handler.load_data()

    # Calculate the current total cases from the data.
    current_infected = data_handler.calculate_total_infected(raw_data)
    print(f" current_infected : {current_infected.shape}")

    # As the time series is non-stationary (the average changes over time) some data tranformations 
    # must take place. This will provide better results.

    # First the natural logarithm of the original data is taken.
    log_cases = pd.DataFrame(np.log(current_infected))
    # Then the difference between each point in the data is calculated.
    first_diff = log_cases.diff().dropna()
    # As the first differencing did not provide good results the difference of the difference is taken.
    stationary_data = first_diff.diff().dropna()
    print(f" stationary_data : {stationary_data.shape}")
    
    # Convert the time series to a supervised learning dataset.
    # In this setting four days (input) of data will be used to forecast the fith day (output).
    supervised_data = series_to_supervised(stationary_data, 0, 4).dropna()
    print(f" supervised_data : {supervised_data.shape}")

    # Split the data into input (x) and output (y).
    x = supervised_data.iloc[:, :4].to_numpy()
    y = supervised_data.iloc[:, 4:].to_numpy()
    print(f" x : {x.shape}")
    print(f" y : {y.shape}")

    # Split the data into train and test sets.
    train_size = int(x.shape[0] * 0.67)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape x from [samples, time steps] to [samples, time steps, features]
    # Where samples = rows and time steps = columns.
    features = 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], features)

    print(f" x_train : {x_train.shape}")
    print(f" x_test : {x_test.shape}")
    print(f" y_train : {y_train.shape}")
    print(f" y_test : {y_test.shape}")

    print("Training...")

    # Define and train the LSTM model.
    model = Sequential()
    model.add(LSTM(10, input_shape=(4, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

    print("Done")

    # Create predictions for the train and test data.
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    print(f" train_prediction : {train_prediction.shape}")
    print(f" test_prediction : {test_prediction.shape}")

#    # Revert the two differening processes applied at the start.
#    train_prediction = reverse_difference(stationary_data[0], train_prediction)
#    train_prediction = reverse_difference(first_diff[0], train_prediction)
#
#    test_prediction = reverse_difference(stationary_data[0], test_prediction)
#    test_prediction = reverse_difference(first_diff[0], test_prediction)
#
#    # Now revert the log by applying the exponential. 
#    # After this the predictions will be on the same scale as the original data.
#    train_prediction = np.exp(train_prediction)
#    test_prediction = np.exp(test_prediction)

    # Calculate the Root Mean Squared Error for the predictions.
    train_error = rmse(y_train, train_prediction)
    test_error = rmse(y_test, test_prediction)

    print(f"RMSE for train data = {train_error}")
    print(f"RMSE for test data = {test_error}")

    train_fig = plt.figure(1) 
    plt.plot(y_train)
    plt.plot(train_prediction)

    test_fig = plt.figure(2) 
    plt.plot(y_test)
    plt.plot(test_prediction)

    plt.show()

