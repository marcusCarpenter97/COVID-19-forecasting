import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date
import data

TRAIN_END = "5/4/2020"  # M/D/YYYY
TRAIN_SIZE = 104  # Number of days since the first day in the timeseries.

# The model will output 'test_offset' days worth of predictions.
# Out of those, 'validation_offset' days will be know and used
# to verify the model's accuracy.
VAL_OFFSET = 21
TEST_OFFSET = 37

COVID_DATA = data.Data()

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
