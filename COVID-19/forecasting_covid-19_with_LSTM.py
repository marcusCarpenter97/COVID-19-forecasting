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

# This function helps print out the data from the results.
def print_results(res, index, row_names):
    formatted_res = pd.Series(res[0:index], index=row_names)
    for key, value in res[index].items():
        formatted_res[f'Critical Value ({key})'] = value
    print(formatted_res)

# ADF
# The null hypothesis for the ADF test is that the time series is not stationary. The objective of this test is to reject (or
# disprove) this hypothesis. Basically, the test assumes that there is a trend (unit root) in the data and the results are used
# to prove or disprove this assumption. To reject the null hypothesis (prove there is no trend) the p-value produced by the test
# must be less than 0.05. The value of 0.05 (alpha) represents the 5% Critical Value (the default option) used in the test. Becasue of
# this the result of the test (Test Statistic) can be compared with the Critical Value at 5% in the same way as the p-value,
# i.e. Test Statistic < Critical Value at 5% = No trend (reject null hypothesis).

# The results will vary based on teh data but the Critical Values are a simple pre-calculated lookup table. If a different alpha
# is use in the test, e.g. 0.10, then the Critical Value at 10% would be used to compare the results.

print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic','p-value','Lags Used','Number of Observations Used'])

# When running the test with 86 observations (days) the Test Statistic was of -0.049970 and the p-value was 0.954199.
# This means that this time series has a unit root as 0.9 is not smaller than 0.05. In other word a trend that can be removed
# through differencing the data.

# KPSS
# The KPSS test complements the ADF test. Its null hypothesis states that the time series is stationary. For example, if the ADF
# accepts its null hypothesis and the KPSS test does not then there is definately a unit root (trend) in the data. However, if 
# the test is not rejected, the KPSS shows that the data is trend stationary. This means that it is stationary around a
# (deterministic) trend.
# For example, a linear trend is present where all values increase over time. If this trend is removed the data will become
# stationary.

# As this is essentialy the opposite of the ADF, the results show that if the p-value is smaller than 0.05 (again the default
# value) then the time series is not stationary.

print("Results for the KPSS test:")
print_results(kpss_results, 3, ['Test Statistic','p-value','Lags Used'])

# When the test was run the Test Statistic was of 1.345703 and the p-value was of 0.010000. As 0.01 is < 0.05 the null hypothesis
# is rejected proving that the data is not stationary.

# This means that to make this time series stationary is through differencing the data as there is no deterministic trend.
