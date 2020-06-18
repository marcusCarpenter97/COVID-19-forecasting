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
HORIZONS = 4  # The number of horizons to forecast.
TEST_SIZE = HORIZON * HORIZONS # Size of the test data depends on the number of horizons.

def print_n_plot_country(country, train_bar=None):
    """ Helper function to display a country's data.
        train_bar is a flag for chosing whether to
        display a line dividing the training and 
        testing sets.
    """
    country.print_country()
    if train_bar:
        country.plot_country(train_date=train_bar)
    else:
        country.plot_country()

COVID_DATA = data.Data()

VOCAB_SIZE, MAX_LENGTH = COVID_DATA.encode_names()

WORLD = COVID_DATA.find_country("World")
print_n_plot_country(WORLD)

COVID_DATA.log()

print_n_plot_country(WORLD)

COVID_DATA.difference()

print_n_plot_country(WORLD)

COVID_DATA.split_train_test(TEST_SIZE, HORIZON)
print(WORLD.train.shape)
print(WORLD.test.shape)

COVID_DATA.supervise_data(HORIZON)
print(WORLD.train_x.shape)
print(WORLD.train_y.shape)

input_shape = (WORLD.train_x.shape[1], WORLD.train_x.shape[2])  # timesteps, features.
output_shape = WORLD.train_y.shape[2] # features.

LSTM = lstm.myLSTM()
LSTM.multivariate_encoder_decoder(input_shape, output_shape, HORIZON)

LSTM.print_summary()
LSTM.plot_model()

LSTM.train(WORLD.train_x, WORLD.train_y)

LSTM.plot_history("multi_ed_LSTM")

def make_predictions(country):
    """
    Create predictions for a country on a weekly basis.

    Parameter:
    country: country obj.

    Returns:
    numpy array containing predictions for country.
    """
    predictions = []
    # Make a prediction for each week in the test data.
    # The last week won't have a ground truth.
    for week in country.test:
        week = week.reshape(1, week.shape[0], week.shape[1])
        predictions.append(LSTM.predict(week))

    predictions = np.stack(predictions)
    # Stacking the arrays produces a 4D array which needs to be reshaped to 2D.
    return predictions.reshape(predictions.shape[0] * predictions.shape[2], predictions.shape[3])

predictions = make_predictions(WORLD)

COVID_DATA.integrate()

def integrate_country_pred(country, predictions):
    """
    Rescales predictions by integrating values.

    Parameters
    country: country obj.
    predictions: numpy array.

    Returns
    DataFrame.
    """
    # Get the original value for the row imediately before the prediction data.
    # This will be used to integrate the predictions.
    # Minus one becasuse of the 0 indexed arrays.
    idx = len(country.data) - len(predictions) - 1
    before_pred = country.data.iloc[idx].values

    def calc_row(before_pred, diffed):
        return before_pred + diffed.sum()
    res = [calc_row(before_pred, predictions[:row]) for row in range(1, len(predictions)+1)]
    return np.stack(res)

int_pred = integrate_country_pred(WORLD, predictions)

COVID_DATA.exp()

exp_pred = np.expm1(int_pred)

# Slit the weeks.
exp_pred = exp_pred.reshape(exp_pred.shape[0]//HORIZON, HORIZON, exp_pred.shape[1])

print(exp_pred)
print(exp_pred.shape)

# Skip last week as there is no gound thruth for it.
pred_weeks = exp_pred[:-1]
# Get the original values. Numper of days = number of weeks * horizon.
orig_weeks = np.array(WORLD.data[-len(pred_weeks)*HORIZON:])
# Make the shape equal to pred_weeks.
orig_weeks = orig_weeks.reshape(orig_weeks.shape[0]//HORIZON, HORIZON, orig_weeks.shape[1])

print(pred_weeks.shape)
print(orig_weeks.shape)

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
