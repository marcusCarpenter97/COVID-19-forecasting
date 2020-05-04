# # Table of contents:

# * 1\. [The data](#data) 
# * 2\. [Objectives](#obj)
#     * 2.1. [Predict current infected](#obj_predcur)
#     * 2.2. [Predict deceased](#obj_preddece)
#     * 2.3. [How well can the LSTM model the problem?](#obj_gener)
#     * 2.4. [How does the LSTM perform when compared with other models?](#obj_comp)
# * 3\. [Time series analysis](#time)
#     * 3.2. [Present the data](#time_data)
#     * 3.3. [Why stationary?](#time_station)
#     * 3.4. [ADF and KPSS tests](#time_tests)
#     * 3.5. [Making the time series stationary](#time_transf)
#          * 3.4.1. [Differencing](#time_transf_diff)
#          * 3.4.2. [Gaussian Curve](#time_transf_gauss)
# * 4\. [Preparing the data for the Neural Network](#prep)
#     * 4.1. [Train and test](#prep_traintest)
#     * 4.2. [Supervised learning](#prep_super)
# * 5\. [The LSTM](#lstm)
#     * 5.1. [Input, hidden and output layers](#lstm_layers)
#     * 5.2. [Training the model](#lstm_train)
#     * 5.3. [Does differencing make a difference?](#lstm_diff)
#     * 5.4. [Results on global data](#lstm_global)
#     * 5.5. [Can it generalize?](#lstm_gener)
# * 6\. [The GRU](#gru)
#     * 6.1. [Input, hidden and output layers](#gru_layers)
#     * 6.2. [Training the model](#gru_train)
#     * 6.3. [Results on global data](#gru_global)
# * 7\. [ARIMA](#arima)
#     * 7.1. [How many integrations?](#arima_int)
# * 8\. [Summary of results](#summary)
#     * 8.1. [Plot predictions of all algorithms over the original data](#summary_global)
#     * 8.2. [Summarize findings](#summary_findings)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduces tensorflow messages.

import warnings
warnings.filterwarnings('ignore')  # Reduces warning messages.

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Activation, Dropout
from keras.callbacks.callbacks import EarlyStopping

import data_handler  # Custom module used to fetch and prepare data from GitHub.

import pandas as pd
import numpy as np

from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]  # Adjust plot sizes.

# # 1. The data <a name="data"></a>

# Before any analysis can be done it is important to understand the data we will be working with. The data comes from the 
# [Johns Hopkins University GitHub repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series). This data set provides three time series; the total infected, recovered and deceased.

# This function fetches the data from the GitHub repository, if necessary, and returns a list of dataframes with each time series.
raw_data = data_handler.load_data()

# Here is how the data looks like in table format.
raw_data[0].head()  # Confirmed.

raw_data[1].head()  # Deceased.

raw_data[2].head()  # Recovered.

# It is also possible to view the data for a specific country.
raw_data[0][raw_data[0]['Country/Region'].isin(['United Kingdom'])]  # Total confirmed cases in the UK.

# The following formula can be used to find the number of current infected as oposed to total infected (confirmed cases):
# current infected = total infected - (deceased + recovered) 
current_infected = data_handler.calculate_current_infected(raw_data)

# Plotting the data hepls to visualise it better. But first all rows must be summed up into one time series.
confirmed = raw_data[0].sum()[2:]  # The first two rows are the sums of Long and Lat which must be removed.
deceased = raw_data[1].sum()[2:]
recovered = raw_data[2].sum()[2:]

fig, ax = plt.subplots()
ax.plot(confirmed)
ax.plot(current_infected)
ax.plot(recovered)
ax.plot(deceased)
ax.set_title('COVID-19 data')
ax.legend(['Total confirmed', 'Current infected', 'Recovered', 'Deceased'])
ax.set_ylabel('People')
ax.set_xlabel('Time')
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 4))
fig.autofmt_xdate()

# # 2. Objectives <a name="obj"></a>

# The Kaggle competition requires two forecasts to be made, the number of infected people and the number of deceased people.
# There are two more objectives proposed in this notebook. One, how well do the machine learning models perform when compared with
# eachother? Two, how good is the LSTM at generalizing the problem of forecasting the COVID-19?

# ## 2.1 The Kaggle compettition <a name="obj_kaggle"></a>

# Now that the data has been explained, it is possible to use any time series forecasting method on it. In this notebook the
# selected methods where the LSTM, GRU and ARIMA. Performance will be compared using the RMSLE as determined by Kaggle.

# ## 2.3 How well can the LSTM model COVID-19? <a name="obj_gener"></a>

# To prove that the LSTM has actualy learnt from the data it is necessary to create forecasts on unseen data. A testing 
# set is usualy separated from the data before using the rest to train a neural network. But how well can the network model the
# virus? This can be verified by using the same model trained on the global data and using it to make predictions on individual
# countries. Technicaly the global data contains all the information present in each individual country. This will allow the
# model to actualy learn how the number of inferted people change over time.

# The following code prepares the data for a plot demonstrating the above pagraph.
uk_data = []
# Total confirmed cases in the UK.
uk_data.append(raw_data[0][raw_data[0]['Country/Region'].isin(['United Kingdom'])])
# Total deceased cases in the UK.
uk_data.append(raw_data[1][raw_data[1]['Country/Region'].isin(['United Kingdom'])])
# Total recovered cases in the UK.
uk_data.append(raw_data[2][raw_data[2]['Country/Region'].isin(['United Kingdom'])])
# Make the deceased table into a time series.
uk_deceased = uk_data[1].sum()[2:]
# Calculate time series of infected people in UK.
uk_infected = data_handler.calculate_current_infected(uk_data)
# Split UK infected.
uk_inf_train, uk_inf_test = data_handler.split_train_test(uk_infected, 0.7)
# Split UK deceased.
uk_dec_train, uk_dec_test = data_handler.split_train_test(uk_deceased, 0.7)
# Split global infected.
global_inf_train, global_inf_test = data_handler.split_train_test(current_infected, 0.7)

fig, ax = plt.subplots()
ax.plot(global_inf_train)
ax.plot(global_inf_test)
ax.plot(uk_inf_test)
ax.plot(uk_dec_test)
ax.set_ylabel('People')
ax.set_xlabel('Time')
ax.legend(['Global infected (train data)', 'Global infected (test data)', 'UK infected (test data)', 'UK deceased (test data)'], loc='best')
# Prevent the x-axis labels from overlapping.
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 4))
fig.autofmt_xdate()

# Is a model trined on the global infected (blue  line) capable of creating accurate forecasts for the UK data?

# ## 2.4 How does the LSTM perform when compared with other models? <a name="obj_comp"></a>

# RMSE, allong with the RMSLE, was used to compare the performance for the models. Plots were also used to show how the selected
# models perform on the data. MASE was used to calculate the accuracy of forecasts. As this uses
# percentages it is possible to compare different methods quite well.
# The three loss functions used to compare the models performance are:

# ### 2.4.1 Root Mean Squared Error (RMSE):
# Calculates the average error in the forecasts. Values calculated by this function represent the number of people the forecast
# got wrong.

# $ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(p_i-r_i)^2} $
# Where *p* is the predicted value, *r* is the real value and *n* is the number of data.
def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())

# ### 2.4.2 Root Mean Squared Log Error (RMSLE):
# Other than being chosen by Kaggle, this loss function penalises predictions smaller than the real value. So, according to this
# function, overestimating the number of infected people is better than underestimating it.

# $ RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\ln(p_i+1)-\ln(r_i+1))^2} $
# Where *p* is the predicted value, *r* is the real value and *n* is the number of data.
def rmsle(prediction, target):
    return np.sqrt(((np.log(prediction+1) - np.log(target+1)) ** 2).mean())

# ### 2.4.3 Mean Absolure Scaled Error (MASE):
# This represents the forecast error as a ratio. This is usefull as it allows forecasts on different scales to be compared.

# $ MASE = \frac{\frac{1}{J}\sum_{j}|p_i-r_i|}{\frac{1}{T-1}\sum_{t=2}^{T}|r_t-r_{t-1}|}  $
# Where *p* is the predicted value, *r* is the real value , *T* is the size of the training set, *J* is the number of forecats.
def mase(prediction, target, train_size):
    mean_error = (abs(prediction - target)).mean()
    scaling_factor = (1/(train_size-1)) * target.diff().abs().sum()
    return float(mean_error / scaling_factor)

# # 3. Time series analysis <a name="time"></a>

# In this chapter an analysis of the time series was performed. This analysis will highlight features in the data which cause the
# time series to have trends. A time series with a trend is called non-stationary because the mean (average) changes over time.
# Once these trends are found they can be removed from the data making it a stationary time series.

# ## 3.1 Why stationary? <a name="time_station"></a>

# To create accurate forecasts the time series must be stationary simplly because it is easier to model data without any trends.
# There are two statistical tests that can be applied to a time series to check whether it is stationary or not. This is better
# than simply 'looking' at the data as it can find (and prove the existence of) useful features in the series.

# The plot of the current infected shows a clear trend in the data.
current_infected.plot()

# ## 3.2 ADF and KPSS tests <a name="time_tests"></a>

# This function helps print out the results from the following tests.
def print_results(res, index, row_names):
    formatted_res = pd.Series(res[0:index], index=row_names)
    for key, value in res[index].items():
        formatted_res[f'Critical Value ({key})'] = value
    print(formatted_res)

# ### 3.2.1 The Augmented Dickey Fuller (ADF) test.

# * The null hypothesis for the ADF test is that the time series is not stationary because it has a unit root.
# * The alternate hypothesis is that there is no unit root in the data making it stationary.
# * The unit root is a feature in a time  series that causes a trend. This can be removed by differencing the data.
# * The null hypothesis is assumed to be true until it is proven to be false, i.e. it is assumed that the data is not stationary by default.
# * To reject the null hypothesis (prove there is no trend) the p-value produced by the test must be less than the Critical Value.
# * The Critical Value represents how certain the test is of its results, e.g. if the p-value is lest than 0.05 (5%) then we can say that the test is 95% confident that the time series is stationary.
adf_results = adfuller(current_infected)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

# As the p-value is larger than all levels of alpha (0.01, 0.05 and 0.1) this means that the time series has a unit root. In fact
# it is possible to multiply the p-value by 100 to see how confident the test is. For example a p-value of 0.973938 means that the
# test is 97% confident that the is a unit root (trend) in the data.

# ### 3.2.2 The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

# * The KPSS test complements the ADF test.
# * Its null hypothesis states that the time series is stationary.
# * While there alternate hypothesis states that the data has a unit root.
# * The p-value and the Critical Values work the same ass the ADF test.
# * However, if the null hypothesis is not rejected, the KPSS shows that the data is 'trend' stationary. This means that it is stationary around a deterministic trend.
# * For example, a linear trend is present where all values increase over time. If this trend is removed the data will become stationary.
kpss_results = kpss(current_infected, nlags='auto')
print("Results for the KPSS test:")
print_results(kpss_results, 3, ['Test Statistic', 'p-value', 'Lags Used'])

# The results show that the null hypothesis is rejected, this means that the unit root is definitely present and that there is no deterministic trend.

# Note: the results will be different every time more data is added to the series.

# ## 3.4 Making the time series stationary <a name="time_transf"></a>

# The tests show there is at least one unit root in the data. This can be removed by differencing the data. As there is no
# deterministic trend in the data there is no need to use the KPSS test again. Another important factor is the range the data is
# in. If the neural network is given large numbers to process, e.g. between 10000 and 50000, it will take too long to train it.
# This range must be 'squashed' into a more  manageable range. However, the method used to transform the data must make it fit in
# a Gaussian curve as it makes data data easier to forecast.

# ### 3.4.1 Differencing <a name="time_transf_diff"></a>

# Differencing takes the difference between each point in the data. The result of this is a series showing the change in the data.
# Example: The fist row shows the number of infected people while the second (differenced) row shows the change in infected
# people. This proces removes the unit root from the data and can be repeated as many times as necessary.
table = current_infected.head().T
table.append(current_infected.head().diff().T)

# It is important to note that every time the data is differenced some level of detail is lost. This can be verified by the NaN
# value in the second row.

# Now that differencing has been explained it can be applied on the full dataset. The graph now shows the global rate of
# infection.
first_differenced = current_infected.diff().dropna()
first_differenced.plot()

# Another ADF test on the differenced data shows that there is still a unit root in the data. This means that the data must be differenced again.
adf_results = adfuller(first_differenced)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

# Differencing the data a second time shows us that even though the graph looks like a stationary time series there still is an unit root in the data as the p-value is greater than all alpha values.
second_differenced = first_differenced.diff().dropna()
second_differenced.plot()
adf_results = adfuller(second_differenced)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

# When differenced a third time the data shows us that there are no more unit roots. The p-value has become so small that there is almost a 100% certainty of it.
third_differenced = second_differenced.diff().dropna()
third_differenced.plot()
adf_results = adfuller(third_differenced)
print("Results for the ADF test:")
print_results(adf_results, 4, ['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

# ### 3.4.2 Gaussian Curve <a name="time_transf_gauss"></a>

# It is desirable that the data fits a Gaussian curve as it is easier to model it. The shape of this dataset can be seen by plotting it as a histogram. It is possible to see that the original data has a long right tail. This means that there a many 'rare values' greater than the mean (average). This makes sense when looking back at the original plot where the curve stays horizontal for a while before the number of infected people start to rise.

# On the other hand, the data after being differenced three times looks more like a Gaussian distribution (the bell shape). This is because the trends where removed.

fig, ax = plt.subplots(2, 1)
current_infected.plot(kind='hist', ax=ax[0])
ax[0].set_title("Original dataset.")
third_differenced.plot(kind='hist', ax=ax[1])
ax[1].set_title("Dataset without unit roots.")
fig.tight_layout()

# #### 3.4.2.1 The BoxCox methods 
one_d_ci = [i[0] for i in current_infected.values]  # The boxcox methods require 1 dimensional data.

data1 = boxcox(one_d_ci, -1)    # reciprocal transform.
data2 = boxcox(one_d_ci, -0.5)  # reciprocal square root transform.
data3 = boxcox(one_d_ci, 0)     # log transform.
data4 = boxcox(one_d_ci, 0.5)   # square root transform.

fig, axes = plt.subplots(2, 2)
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

# From the graphs above it is possible to see that the method that makes the data look more like a Gaussian distribution is the
# log transform. This can be used on the data to 'squash' the values before giving it to the neural network. This is important
# because it would take longer to process the larger numbers. Taking the log of the data will reduce training time without
# drastically affecting the structure of the data.

# # 4. Preparing the data for the Neural Network <a name="prep"></a>

# ## 4.1 Train and test <a name="prep_traintest"></a>

# The neural networks used in this experiment are the LSTM (a type of RNN) and the GRU (a simplified LSTM). These were chosen as
# they were designed to handle sequence data like a time series.

forecast_horizon = 4   # Number of observations to be used to predict the next event.
train_set_ratio = 0.7  # The size of the training set as a percentage of the data set.

# Split the data into train and test sets. This must be done before any transformations to avoid data leakage. Otherwise there might be information about the test set in the train set because of the data transformation applied.
# Note: the transformations done before where for the analysis of the time series, now they will be used to actually reshape the data.
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
train_log, train_diff_one, train_diff_two, stationary_train = data_handler.adjust_data(train)
test_log, test_diff_one, test_diff_two, stationary_test = data_handler.adjust_data(test)

# ## 4.2 Supervised learning <a name="prep_super"></a>

# Transform the data into a supervised learning dataset. Essentially the neural network will use a number of observations
# (forecast horizon) to predict the next event in the sequence, e.g. use four days of data to predict the fifth.
supervised_train = data_handler.series_to_supervised(stationary_train, 0, forecast_horizon)
supervised_test = data_handler.series_to_supervised(stationary_test, 0, forecast_horizon)

# Create sets for input and output based on the forecast horizon.
x_train, y_train = data_handler.split_horizon(supervised_train, forecast_horizon)
x_test, y_test = data_handler.split_horizon(supervised_test, forecast_horizon)

# This part is required by the Keras models as they take 3 dimensional input.
# Reshape x from [samples, time steps] to [samples, time steps, features]
# Where samples = rows and time steps = columns.
features = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], features)

# # 5. The LSTM <a name="lstm"></a>

# A standard sequential model. One layer after the other.
lstm_model = Sequential()

# ## 5.1 Input, hidden and output layers <a name="lstm_layers"></a>

# ### 5.1.1 The input layer.

# The LSTM receives the input as an array of size equals to the forecast horizon where each item is an array of size one.
# e.g. [[1], [2], [3], [4]]
input_layer = (forecast_horizon, 1)

# ### 5.1.2 The hidden layer.

nodes = 5  #  The number of nodes in the hidden layer represents how 'wide' the network will be.
lstm_model.add(LSTM(nodes, input_shape=input_layer))

lstm_activation = "relu"  # The activation function decides how much information flows though the nodes.
lstm_model.add(Activation(lstm_activation))

# The dropout function selects nodes at random and uses them during one training epoch (iteration). This is applied for each training epoch.
#This helps reduce overfitting by making the nodes learn the data instead of relying on other nodes.
rate = 0.1 # The probability of a node being ignored during training, e.g. 10%.
lstm_model.add(Dropout(rate))

# ### 5.1.3 The output layer.

out_shape = 1  # The number of outputs the LSTM will produce. Technically the number of nodes in the dense layer.
# It is possible to output the forecasts for the next few days but in this model only one value will be produced.
lstm_model.add(Dense(out_shape))

dense_activation = "sigmoid"  # The activation function for the output node works the same as for the hidden layer.
lstm_model.add(Activation(dense_activation))

# ## 5.2 Training the model <a name="lstm_train"></a>

# Here the model is told how to learn. The objective of the optimizer is to make the output of the loss function smaller.
loss = "mean_squared_error"
opti = "adam"
lstm_model.compile(loss=loss, optimizer=opti)

# This will stop the training if the loss does not change for 50 consecutive epochs.
early_stopping = EarlyStopping(patience=50, restore_best_weights=True)

# Here the model is trained. As this was framed as a supervised learning task the LSTM will try to match the input (x_train) to
# the output (y_train). One input will pass through the LSTM at a time (batch_size) until all inputs are processed. This will be
# done 1000 times (epochs) and in each iteration the network will (hopefully) 'learn' a little more. One fifth of the training
# data (0.2) is left out as a validation set for each epoch.
lstm_history = lstm_model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0, callbacks=[early_stopping],
                    validation_split=0.2)

# ## 5.3 Does differencing make a difference? <a name="lstm_diff"></a>
# ## 5.4 Results on global data <a name="lstm_global"></a>

# Make the predictions. The predictions on the training data are only used to measure the model's performance as it should have
# already learnt it.
train_prediction = lstm_model.predict(x_train)
test_prediction = lstm_model.predict(x_test)

# Rescale predictions.
scaled_train_predictions = data_handler.rescale_data(train_prediction, 
                                                     train_diff_one[0], 
                                                     train_diff_two[0], 
                                                     train_log[0], 
                                                     forecast_horizon)
scaled_test_predictions = data_handler.rescale_data(test_prediction, 
                                                    test_diff_one[0], 
                                                    test_diff_two[0], 
                                                    test_log[0], 
                                                    forecast_horizon)

# Rescale the answers.
scaled_train = data_handler.rescale_data(y_train, 
                                         train_diff_one[0], 
                                         train_diff_two[0], 
                                         train_log[0], 
                                         forecast_horizon)
scaled_test = data_handler.rescale_data(y_test, 
                                        test_diff_one[0], 
                                        test_diff_two[0], 
                                        test_log[0], 
                                        forecast_horizon)

# Calculate the error values.
lstm_train_rmse = rmse(scaled_train, scaled_train_predictions)
lstm_test_rmse = rmse(scaled_test, scaled_test_predictions)
lstm_train_rmsle = rmsle(scaled_train, scaled_train_predictions)
lstm_test_rmsle = rmsle(scaled_test, scaled_test_predictions)

# Plot model loss history.
fig, ax = plt.subplots()
ax.plot(lstm_history.history['loss'])
ax.plot(lstm_history.history['val_loss'])
ax.set_title('Model loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(['Train', 'Test'], loc='best')

# Some preparation to plot the predictions is needed.

# The first sample is lost after each differencing, so the + 2 is required. 
empty_arr = np.empty((forecast_horizon+2, 1))
empty_arr[:] = np.nan
shifted_train = np.concatenate([empty_arr, scaled_train_predictions])
# The test data mus be shifted by 2 empty arrays plus the training data.
empty_arr = np.empty(((forecast_horizon+2)*2+len(scaled_train_predictions), 1))
empty_arr[:] = np.nan
shifted_test = np.concatenate([empty_arr, scaled_test_predictions])

# Plot the predictions over the original dataset.
fig, ax = plt.subplots()
ax.plot(current_infected)
ax.plot(shifted_train)
ax.plot(shifted_test)
ax.set_title('Prediction over original')
ax.set_ylabel('Number of infected')
ax.set_xlabel('Time')
ax.legend(['Original data', 'Train predictions', 'Test predictions'], loc='best')
# Prevent the x-axis labels from overlapping.
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 4))
fig.autofmt_xdate()

print(f"RMSE on train: {lstm_train_rmse}")
print(f"RMSE on test: {lstm_test_rmse}")
print(f"RMSLE on train: {lstm_train_rmsle}")
print(f"RMSLE on test: {lstm_test_rmsle}")

# ## 5.5 Can it generalize? <a name="lstm_gener"></a>

# # 6. The GRU <a name="gru"></a>

gru_model = Sequential()

# ## 6.1 Input, hidden and output layers <a name="gru_layers"></a>

# ### 6.1.1 The input layer.

gru_input_layer = (forecast_horizon, 1)

# ### 6.1.2 The hidden layer.

gru_nodes = 5
gru_model.add(GRU(gru_nodes, input_shape=gru_input_layer))

gru_activation = "relu"
gru_model.add(Activation(gru_activation))

gru_rate = 0.1
gru_model.add(Dropout(gru_rate))

# ### 6.1.3 The output layer.

gru_out_shape = 1
gru_model.add(Dense(gru_out_shape))

gru_dense_activation = "sigmoid"
gru_model.add(Activation(gru_dense_activation))

# ## 6.2 Training the model <a name="gru_train"></a>

gru_loss = "mean_squared_error"
gru_opti = "adam"
gru_model.compile(loss=gru_loss, optimizer=gru_opti)

# The early stopping callbak is the same as the one used in the LSTM.
gru_history = gru_model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0, callbacks=[early_stopping],
                            validation_split=0.2)

# ## 6.3 Results on global data <a name="gru_global"></a>

gru_train_prediction = gru_model.predict(x_train)
gru_test_prediction = gru_model.predict(x_test)
 
gru_scaled_train_predictions = data_handler.rescale_data(gru_train_prediction, 
                                                     train_diff_one[0], 
                                                     train_diff_two[0], 
                                                     train_log[0], 
                                                     forecast_horizon)
gru_scaled_test_predictions = data_handler.rescale_data(gru_test_prediction, 
                                                    test_diff_one[0], 
                                                    test_diff_two[0], 
                                                    test_log[0], 
                                                    forecast_horizon)

gru_scaled_train = data_handler.rescale_data(y_train, 
                                         train_diff_one[0], 
                                         train_diff_two[0], 
                                         train_log[0], 
                                         forecast_horizon)
gru_scaled_test = data_handler.rescale_data(y_test, 
                                        test_diff_one[0], 
                                        test_diff_two[0], 
                                        test_log[0], 
                                        forecast_horizon)

gru_train_rmse = rmse(gru_scaled_train, gru_scaled_train_predictions)
gru_test_rmse = rmse(gru_scaled_test, gru_scaled_test_predictions)
gru_train_rmsle = rmsle(gru_scaled_train, gru_scaled_train_predictions)
gru_test_rmsle = rmsle(gru_scaled_test, gru_scaled_test_predictions)

print(f"RMSE on train: {gru_train_rmse}")
print(f"RMSE on test: {gru_test_rmse}")
print(f"RMSLE on train: {gru_train_rmsle}")
print(f"RMSLE on test: {gru_test_rmsle}")

# # 7. ARIMA <a name="arima"></a>
# ## 7.1 How many integrations? <a name="arima_int"></a>

# # 8. Summary of results <a name="summary"></a>
# ## 8.1 Plot predictions of all algorithms over the original data <a name="summary_global"></a>
# ## 8.2 Summarize findings <a name="summary_findings"></a>
