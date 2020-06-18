import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
import data
import lstm

HORIZON = 7  # Number of days in a horizon.
HORIZONS = 4  # The number of horizons to forecast.
TEST_SIZE = HORIZON * HORIZONS # Size of the test data depends on the number of horizons.

COUNTRY_NAME = "World"  # For testing.

COVID_DATA = data.Data()

VOCAB_SIZE, MAX_LENGTH = COVID_DATA.encode_names()

COVID_DATA.print_n_plot_country(COUNTRY_NAME)

COVID_DATA.log()

COVID_DATA.print_n_plot_country(COUNTRY_NAME)

COVID_DATA.difference()

COVID_DATA.print_n_plot_country(COUNTRY_NAME)

COVID_DATA.split_train_test(TEST_SIZE, HORIZON)

COVID_DATA.supervise_data(HORIZON)

# The first country in the data is used to define the input and output shapes.
# This assumes that all countries have the same shape.
input_shape = (COVID_DATA.countries[0].train_x.shape[1], COVID_DATA.countries[0].train_x.shape[2])  # timesteps, features.
output_shape = COVID_DATA.countries[0].train_y.shape[2] # features.

LSTM = lstm.myLSTM()
LSTM.multivariate_encoder_decoder(input_shape, output_shape, HORIZON)

LSTM.print_summary()
LSTM.plot_model()

# Train and test on World data.
WORLD = COVID_DATA.find_country("World")

LSTM.train(WORLD.train_x, WORLD.train_y)

LSTM.plot_history("multi_ed_LSTM")

predictions = LSTM.make_predictions(WORLD)

COVID_DATA.integrate()

int_predictions = WORLD.int_pred(predictions)

COVID_DATA.exp()

exp_pred = np.expm1(int_predictions)

# Slit the weeks.
exp_pred = exp_pred.reshape(exp_pred.shape[0]//HORIZON, HORIZON, exp_pred.shape[1])

# Skip last week as there is no gound thruth for it.
pred_weeks = exp_pred[:-1]
# Get the original values. Numper of days = number of weeks * horizon.
orig_weeks = np.array(WORLD.data[-len(pred_weeks)*HORIZON:])
# Make the shape equal to pred_weeks.
orig_weeks = orig_weeks.reshape(orig_weeks.shape[0]//HORIZON, HORIZON, orig_weeks.shape[1])

weekly_errors = []
# Calculate RMSE for each feature in each week.
for pred_week, orig_week in zip(pred_weeks, orig_weeks):
    feature_errors = []
    # Transpose array to iterate through columns which represents a feature.
    for pred_feature, orig_feature in zip(pred_week.T, orig_week.T):
        feature_errors.append(LSTM.rmse(pred_feature, orig_feature))
    weekly_errors.append(feature_errors)

print(weekly_errors)
plt.show()
