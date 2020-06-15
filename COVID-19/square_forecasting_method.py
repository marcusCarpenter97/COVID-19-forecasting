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
print(predictions)
print(predictions.shape)

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
    return pd.DataFrame(res)

ans = integrate_country_pred(WORLD, predictions)
print(ans)
print(ans.shape)
#plt.show()
raise SystemExit

COVID_DATA.exp()

ans = np.expm1(ans)

print(ans)

# The test set is bigger than the predictions
# so only the part that overlaps with the test
# is used to calculate the errors.
RMSE = LSTM.rmse(ans.values, WORLD.data[-len(ans):].values)

print(RMSE)
plt.show()
# TODO: dont use Healthy
# TODO: make data in shape w1 -> w2 etc...
# TODO: implement encoder decoder LSTM for weeks.
