import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
import data
import LSTM

TRAIN_END = "5/4/20"  # M/D/YY TODO remove
TRAIN_SIZE = 104  # Number of days since the first day in the timeseries.

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
COVID_DATA.difference()
print_n_plot_country(WORLD)
plt.show()
