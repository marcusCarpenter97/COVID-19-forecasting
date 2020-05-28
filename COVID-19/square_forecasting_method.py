import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
import data

TRAIN_END = "5/4/20"  # M/D/YY TODO remove
TRAIN_SIZE = 104  # Number of days since the first day in the timeseries.

# The model will output 'test_offset' days worth of predictions.
# Out of those, 'validation_offset' days will be know and used
# to verify the model's accuracy.
VAL_OFFSET = 21
TEST_OFFSET = 37

def print_n_plot():
    while True:
        while True:
            COUNTRY_NAME = input("Enter a country's name: ")
            if COUNTRY_NAME == "exit":
                raise SystemExit
            COUNTRY = COVID_DATA.find_country(COUNTRY_NAME)
            if COUNTRY:
                break

        COUNTRY.print_country()

        COVID_DATA.plot_data(COUNTRY_NAME)

        # This stops the plots from closing automaticaly.
        plt.show()

COVID_DATA = data.Data()

# # Objectives

# # Data

# ## Plot global data.

def print_n_plot_global():
    """ Helper function to display the global data. """
    print("Global data.")
    COVID_DATA.print_global()
    COVID_DATA.plot_global()

#print_n_plot_global()

# ## Plot relevant countries.

def print_n_plot_country(country, train_bar=False):
    """ Helper function to display a country's data.
        train_bar is a flag for chosing whether to
        display a line dividing the training and 
        testing sets.
    """
    print(f"Data for: {country.name}.")
    country.print_country()
    if train_bar:
        country.plot_country(train_date=TRAIN_SIZE)
    else:
        country.plot_country()

US = COVID_DATA.find_country("US")
UK = COVID_DATA.find_country("United Kingdom")
CHINA = COVID_DATA.find_country("China")
BRAZIL = COVID_DATA.find_country("Brazil")

print("US data.")
#print_n_plot_country(US)

print("UK data.")
#print_n_plot_country(UK)

print("China data.")
#print_n_plot_country(CHINA)

print("Brazil data.")
#print_n_plot_country(BRAZIL)

# ## Discuss data. (total vs rate, all columns)

# # Model evaluation

# ## Error functions.

# ## Compare with GRU.

# # Time series analysis

# ## Stationary and tests.

# ## BoxCox test and method.

# # Data for the NN

COVID_DATA.difference()

#print_n_plot_country(US)
#print_n_plot_country(UK)
#print_n_plot_country(CHINA)
#print_n_plot_country(BRAZIL)

# ## TRAIN_END ...
print_n_plot_country(UK, train_bar=True)

# ## Forecasting the horizon.

# # The models

# ## LSTM.

# ## GRU.

# # Results

# ## Performance on global data.

# ## Performance on countries.

# ## Overall performance.

# ## Compare with other methods.

plt.show()
