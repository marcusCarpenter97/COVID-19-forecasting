import data_handler
import pandas as pd
import numpy as np
from scipy.stats import boxcox
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller, kpss
plt.rcParams['figure.figsize'] = [10, 5]

# Load the time series data of total infected people.
raw_data = data_handler.load_data()
current_infected = data_handler.calculate_total_infected(raw_data)

# Analysing the time series.

# Plot the time series.
current_infected.plot()

# To create accurate forecasts the time series must be stationary. In other words the average must be the same throughout the whole time series. From the original plot it is possible to see that the data is not stationary as the trend seems to be increasing.
# There are two statistical tests that can be applied to a time series to check whether it is stationary or not. This is better than simply 'looking' at the data.

# First, the Augmented Dickey Fuller (ADF) test. This will check if the time series is stationary or if it requires differencing.
adf_results = adfuller(current_infected)

# Second, the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. This checks whether there is a trend in the data, e.g. the values are gradually increasing.
kpss_results = kpss(current_infected, nlags='auto')

# This function helps print out the data from the results.
def print_results(res, index, row_names):
    formatted_res = pd.Series(res[0:index], index=row_names)
    for key, value in res[index].items():
        formatted_res[f'Critical Value ({key})'] = value
    print(formatted_res)

# ADF
# The null hypothesis for the ADF test is that the time series is not stationary because it has a unit root.
# The alternate hypothesis is that there is no unit root in the data making it stationary.
# The unit root is a  feature in a time  series that causes a trend. This can be removed by differencing the data.
# The null hypothesis is assumed to be true until it is proven to be false, i.e. it is assumed that the data is not stationary by
# default.
# To reject the null hypothesis (prove there is no trend) the p-value produced by the test must be less than the Critical Value.
# The Critical Value represents how certain the test is of its results, e.g. if the p-value is lest than 0.05 (5%) then we can say that the test is 95% confident that the time series is stationary.

print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic','p-value','Lags Used','Number of Observations Used'])

# As the p-value is larger than all levels of alpha (0.01, 0.05 and 0.1) this means that the time series has a unit root. In fact
# it is possible to multiply the p-value by 100 to see how confident the test is. For example a p-value of 0.973938 means that the
# test is 97% confident that the is a unit root (trend) in the data.

# KPSS
# The KPSS test complements the ADF test.
# Its null hypothesis states that the time series is stationary.
# While there alternate hypothesis states that the data has a unit root.
# The p-value and the Critical Values work the same ass the ADF test.
# However, if the null hypothesis is not rejected, the KPSS shows that the data is 'trend' stationary. This means that it is stationary around a deterministic trend.
# For example, a linear trend is present where all values increase over time. If this trend is removed the data will become
# stationary.

print("Results for the KPSS test:")
print_results(kpss_results, 3, ['Test Statistic','p-value','Lags Used'])

# The results show that the null hypothesis is rejected, this means that the unit root is defiantly present and that there is no deterministic trend.

# Making the time series stationary.

# As the tests show there is at least one unit root  in the data.  This can be removed through differencing the time series.
# Essentially the difference is taken between each point in the data. As there is no deterministic trend in the data there is  not
# need to use the KPSS test again.
first_differenced = current_infected.diff().dropna()
first_differenced.plot()

# Another ADF test on the differenced data shows that there is still  a unit root in the data. This means that the data must be 
# differenced again.
adf_results = adfuller(first_differenced)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic','p-value','Lags Used','Number of Observations Used'])

# Differencing the data a second time shows us that even though the graph looks line a stationary time series there still is a
# unit root in the data as the p-value is greater than all alpha values.
second_differenced = first_differenced.diff().dropna()
second_differenced.plot()
adf_results = adfuller(second_differenced)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic','p-value','Lags Used','Number of Observations Used'])

# When differenced a third time the data shows us that there is not more unit roots. The p-value has become so small that there is almost a 100% certainty of it.
third_differenced = second_differenced.diff().dropna()
third_differenced.plot()
adf_results = adfuller(third_differenced)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic','p-value','Lags Used','Number of Observations Used'])

# The Gaussian Curve.

#  It is  desirable that the data fits a Gaussian curve as it is easier to model it. The shape of this dataset can be seen by
#  plotting it as a histogram. It is possible to see that the original data has a long right tail. This means that there a many
#  'rare values' greater than the mean (average). This makes sense when looking back at the original plot where the curve stays
#  horizontal for a while before the number of infected people start to rise.

# On the other hand, the data after being differenced three times looks more like a Gaussian distribution (the bell shape). This is because the trends where removed.

fig, ax = plt.subplots(2,1)
current_infected.plot(kind='hist', ax=ax[0])
ax[0].set_title("Original dataset as a histogram.")
third_differenced.plot(kind='hist', ax=ax[1])
ax[1].set_title("Dataset without unit roots as a histogram.")
fig.tight_layout()

one_d_ci = [i[0] for i in current_infected.values]  # The boxcox methods require 1 dimensional data.

data1 = boxcox(one_d_ci, -1)    # reciprocal transform.
data2 = boxcox(one_d_ci, -0.5)  # reciprocal square root transform.
data3 = boxcox(one_d_ci, 0)     # log transform.
data4 = boxcox(one_d_ci, 0.5)   # square root transform.

fig, axes = plt.subplots(2,2)
axes[0,0].hist(data1)
axes[0,0].set_title('Reciprocal transform')
start, end = axes[0,0].get_xlim()  # Prevents the labels on the x-axis from overlapping.
axes[0,0].xaxis.set_ticks(np.arange(start, end, 0.00050))
axes[0,1].hist(data2)
axes[0,1].set_title('Reciprocal square root transform')
axes[1,0].hist(data3)
axes[1,0].set_title('Log transform')
axes[1,1].hist(data4)
axes[1,1].set_title('Square root transform')
fig.tight_layout()

# From the graphs above it is  possible to see that the method that makes the data look like a Gaussian distribution is the log
# transform. This can be used on the data to 'squash' the values before giving it to the neural network. This is important
# because it would take longer to process the larger numbers. Taking the log of the data will reduce training time without
# affecting the structure of the data by too much.

# Preparing the data for the Neural Network.

# The neural network used is the LSTM which is a type of RNN. This was chose as it was designed to handle sequences like a time
# series.

forecast_horizon = 4   # Number of observations to be used to predict the next event.
train_set_ratio = 0.7  # The size of the training set as a percentage of the data set.

# Split the data into train and test sets. This must be done before any transformations to avoid data leakage. Otherwise there
# might be information about the test set in the train set because of the data transformation applied.
train, test = data_handler.split_train_test(current_infected, train_set_ratio)

# This helps visualize the train and test data.
fig, ax = plt.subplots()
ax.plot(train)
ax.plot(test)
ax.set_ylabel('Number of infected')
ax.set_xlabel('Time')
ax.legend(['Train data', 'Test data'], loc='best')
# Prevent the x-axis labels from overlapping.
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 4))
fig.autofmt_xdate()

# Make the time series data stationary based on what we found before.
train_log, train_diff, stationary_train = data_handler.adjust_data(train)
test_log, test_diff, stationary_test = data_handler.adjust_data(test)

# Transform the data into a supervised learning dataset. Essentially the LSTM will use a number of observations
# (forecast horizon) to predict the next event in the sequence, e.g. use four days of data to predict the fifth.
supervised_train = data_handler.series_to_supervised(stationary_train, 0, forecast_horizon)
supervised_test = data_handler.series_to_supervised(stationary_test, 0, forecast_horizon)

# Create sets for input and output based on the forecast horizon.
x_train, y_train = data_handler.split_horizon(supervised_train, forecast_horizon)
x_test, y_test = data_handler.split_horizon(supervised_test, forecast_horizon)

# This part is required by the LSTM as it only takes 3 dimensional input.
# Reshape x from [samples, time steps] to [samples, time steps, features]
# Where samples = rows and time steps = columns.
features = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], features)
