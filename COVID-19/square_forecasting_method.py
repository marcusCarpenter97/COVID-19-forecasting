import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
import data
import lstm

TRAIN_END = "5/4/20"  # M/D/YY TODO remove

# Values chosen to fit the data.
# Might break model if changed without care.
HORIZON = 7  # Number of days in a horizon.
TRAIN_SIZE = 105  # Number of days to be included in the train set.

# The model will output 'test_offset' days worth of predictions.
# Out of those, 'validation_offset' days will be know and used
# to verify the model's accuracy.
VAL_SIZE = 25
OUTPUT_SIZE = 38

def print_n_plot_global():
    """ Helper function to display the global data. """
    print("Global data.")
    COVID_DATA.print_global()
    COVID_DATA.plot_global()

def print_n_plot_country(country, train_bar=None):
    """ Helper function to display a country's data.
        train_bar is a flag for chosing whether to
        display a line dividing the training and 
        testing sets.
    """
    print(f"Data for: {country.name}.")
    country.print_country()
    if train_bar:
        country.plot_country(train_date=train_bar)
    else:
        country.plot_country()

COVID_DATA = data.Data()

WORLD = COVID_DATA.find_country("World")
print_n_plot_country(WORLD)

COVID_DATA.log()

print_n_plot_country(WORLD)

COVID_DATA.difference()

print_n_plot_country(WORLD)

COVID_DATA.split_train_test(HORIZON, TRAIN_SIZE)
print(WORLD.train.shape)
print(WORLD.test.shape)

COVID_DATA.supervise_data(HORIZON)
print(WORLD.train_x.shape)
print(WORLD.train_y.shape)

input_shape = (WORLD.train_x.shape[1], WORLD.train_x.shape[2])  # timesteps, features.
output_shape = WORLD.train_y.shape[1] # features.

LSTM = lstm.myLSTM()
LSTM.multivariate_encoder_decoder(input_shape, output_shape)

LSTM.print_summary()

LSTM.train(WORLD.train_x, WORLD.train_y)

LSTM.plot_history("uni_ed_LSTM")

# Make predictions.
predictions = []
for offset in range(len(WORLD.test)):
    idx = HORIZON+offset
    if idx < len(WORLD.test):
        target = WORLD.test[idx]
        pred_in = WORLD.test[offset:idx]
        pred_in = pred_in.reshape(1, HORIZON, pred_in.shape[1])
        predictions.append(LSTM.predict(pred_in))

predictions = np.stack(predictions)
print(predictions.shape)
print(predictions)

plt.show()
