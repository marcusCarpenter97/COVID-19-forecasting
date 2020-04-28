import data_handler
import pandas as pd
import numpy as np
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller, kpss

# Load the time series data of total infected people.
raw_data = data_handler.load_data()
current_infected = data_handler.calculate_total_infected(raw_data)

# Plot the time series.
current_infected.plot()

# To create accurate forecasts the time series must be stationary. In other words the averege must be the same throughout the whole time series. From the original plot it is possible to see that the data is not stationary as the trend seems to be increasing.
# There are two statistical tests that can be applied to a time series to check whether it is stationary or not. This is better than simply 'looking' at the data.

# First, the Augmented Dickey Fuller (ADF) test. This will check if the time series is stationary or if it requires differencing.
adf_results = adfuller(current_infected)

# Second, the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. This checks whether there is a trend in the data, e.g. the values are gradualy increasing.
kpss_results = kpss(current_infected, nlags='auto')

#This function helps print out the data from the results.
def print_results(res, index, row_names):
    formatted_res = pd.Series(res[0:index], index=row_names)
    for key, value in res[index].items():
        formatted_res[f'Critical Value ({key})'] = value
    print(formatted_res)

# Now we display the test results and interpret them.
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic','p-value','Lags Used','Number of Observations Used'])
#To prove that there is a trend the result must be smaller than (more negative) the Critical Value.

#The ADF test gives a Test Statistic of -0.054291. This means that the data, according to the ADF test, is not stationary because the result is greater than all the Critical Values.

#KPSS
#In the KPSS test the Test Statistic is 0.676763. One of the four critcal values is greater than the test values. This makes it stationary?
print("\nResults for the KPSS test:")
print_results(kpss_results, 3, ['Test Statistic','p-value','Lags Used'])

#  The ADF show that it is not stationary becasue the test statistic it greater than the all the critical values.
# In the KPSS, One of the four critcal values is greater than the test values. This makes it stationary?
