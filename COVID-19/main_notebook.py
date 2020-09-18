# # Table of contents:

# * 1\. [Introduction](#1)
# * 1.1 [Code initialization](#1.1)
# * 2\. [Data](#2)
# * 2.1 [Calculating extra columns](#2.1)
# * 2.2 [Example data](#2.2)
# * 3\. [Time series](#3)
# * 3.1 [Autocorrelation](#3.1)
# * 3.2 [Testing for trends](#3.2)
# * 3.2.1 [Augmented Dickey Fuller](#3.2.1)
# * 3.2.2 [Kwiatkowski-Phillips-Schmidt-Shin](#3.2.2)
# * 3.3 [The Box-Cox test](#3.3)
# * 4\. [Measuring performance](#4)
# * 5\. [Framing the problem](#5)
# * 6\. [The model](#6)
# * 7\. [Experiments](#7)
# * 8\. [Analysing the results](#8)
# * 9\. [Comparrison with previous experiments](#9)
# * 10\. [Conclusion](#10)

# # 1. Introduction <a name="1"></a>

# The main objective of this work is to produce long range forecasts for the number of COVID-19 cases for a given country.

# ## 1.1 Code initialization <a name="1.1"></a>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
COVID_DATA = data.Data()

# # 2. Data <a name="2"></a>

# The data was acquired from the [Johns Hopkins University GitHub repository](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series).
# This dataset contains three files, confirmed, deaths and recovered, each containg time series data for 188 regions. All time
# series start on the 22/01/2020 and, at the time of writing, there is no end date as new data is constantly being appended to
# it.

# ## 2.1 Calculating extra columns <a name="2.1"></a>

# The number of confirmed cases does not accurately represent the current situation of the disease as it will always rise. For a
# more accurate representation the number of currently infected people must be used. As this value is not provided in the data
# it was calculated from the known values for each day. The formula for the current infected is:

# $ Infected = Confirmed - (Deceased + Recovered) $

# If the population of each country is known it is possible to calculate the amount of healthy people from the data once the
# Infected is caculated. This can be done with the formula:

# $ Healthy =  Population - (Infected + Deceased + Recovered) $

# ## 2.2 Example data <a name="2.2"></a>

# Here the first and last five rows of the global data are printed along with its plot. The data for the whole word can be
# calculated by summing all individual countries into one.
COVID_DATA.print_n_plot_country("World")

# The Healthy column is not included in the plot as it would overshadow the other values.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(COVID_DATA.find_country("World").data.values)

fig, ax = plt.subplots(2, 4, figsize=(15,5))
COVID_DATA.find_country("World").data["Confirmed"].plot(title="Confirmed", kind="hist", ax=ax[0,0])
COVID_DATA.find_country("World").data["Deceased"].plot(title="Deceased", kind="hist", ax=ax[0,1])
COVID_DATA.find_country("World").data["Recovered"].plot(title="Recovered", kind="hist", ax=ax[0,2])
COVID_DATA.find_country("World").data["Infected"].plot(title="Infected", kind="hist", ax=ax[0,3])
ax[1,0].hist(scaled_data[:,0])
ax[1,0].set_title("Confirmed (Scaled)")
ax[1,1].hist(scaled_data[:,1])
ax[1,1].set_title("Deceased (Scaled)")
ax[1,2].hist(scaled_data[:,2])
ax[1,2].set_title("Recovered (Scaled)")
ax[1,3].hist(scaled_data[:,3])
ax[1,3].set_title("Infected (Scaled)")
fig.tight_layout()

# The plot above shows a comparison between the original data (top row) and its standarized version (bottom row). This process 
# used the formula $ y = (x-u)/s $ where y is the standarrizes data, x is the original, u represents the mean and s is the 
# standard deviation.

# Standardizing the data does not change its distribution as it simply makes the mean of the data zero and the standard 
# deviation 1. It is important to note that to avoid data leakeage the train and test sets must be split before applying 
# standarization on the data.

# # 3. Time series <a name="3"></a>

# ## 3.1 Autocorrelation <a name="3.1"></a>

# When making predictions based on a time series it is important to check whether the data is random or not. If the time series
# happens to be random then no model would be able to make a accurate prediction. This can be tested with an autocorrelation
# plot. Essencialy this compares how similar the data is to it self by shifting it by a certain amount. These shifts are called
# lags. When the autocorrelation plot shows a value of 1 the time series is identical to its lagged version, if the plot shows a
# 0 then the values are random. Any values outside of the shaded area are considered to be statistically significant.

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Confirmed"], title="Confirmed", ax=ax[0,0])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Deceased"], title="Deceased", ax=ax[0,1])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Recovered"], title="Recovered", ax=ax[1,0])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Infected"], title="Infected", ax=ax[1,1])
fig.tight_layout()

# The autocorrelation plots show that the World data is not random as there is a clear pattern in the plots. This finding is
# generalized to conclude that all countries exibit the same behaviour (strong autocorrelation) as the World data is the sum of
# all individual regions.

# ## 3.2 Testing for trends <a name="3.2"></a>

# Time series data can have trends in them which can make forecasting them mode complex. Therefore these trends are usually
# removed from the data before giving it to the forecasting model. There can be two types of trend, a unit root or a
# deterministic trend.

# A unit root causes the time series to deviate from its mean in a way that it may never return to the same patern as before.
# This type of trend can be removed from the data by a process called differencing which can be applied as many times as
# necessary. Differencing consists of taking the difference between each point and its predecessor. The new time series
# resulting from this process will essentially represent the rate of chence of the original data. For example, a time series of
# the total number of infected people, if differenced, will now show the numbers of newlly infected people at each timestep.

# If a time series has a deterministic trend then the mean of the data will follow it no matter what. There might be small
# perturbation mmaking the values on the y-axis go up or down, however, the mean will always follow the trend.

# There are two tests that check for trends in a time series. The Augmented Dickey Fuller (ADF) and the
# Kwiatkowski-Phillips-Schmidt-Shin (KPSS). These statistical tests complement each other in the sence that the ADF checks for
# the unot root while the KPSS verifies the presence of a trend. Both tests use the null hypothesis to prove or disprove the
# presence of a trend.

def print_results(res, index, row_names):
    """
    This function helps print the results from both the ADF and the KPSS tests.
    """
    formatted_res = pd.Series(res[0:index], index=row_names)
    for key, value in res[index].items():
        formatted_res[f'Critical Value ({key})'] = value
    print(formatted_res)

# ### 3.2.1 Augmented Dickey Fuller (ADF) <a name="3.2.1"></a>

# The ADF test states that the null hypothesis is that the time series has a unit root (not stationary). The alternate
# hypothesis states that the time series is stationary as there is no trend. By default the null hypothesis is asumed to be
# true. The test attempts to disprove this by calculating a p-value which must be less than a Critical value.

# Critical values are set percentages essentialy a lookup table of numbers. The p-value represents the confidence in which the
# null hypothesis is rejected.

adf_results = adfuller(COVID_DATA.find_country("World").data["Confirmed"])
print_results(adf_results, 4, ['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

# The ADF test was applied to the Confirmed cases for the World. The results show a p-values of 0.98 meaning that it is 98% 
# confident that there is a unit root (trend) present in the data. This is noticeable by looking at its plot in section 2.2.

# ### 3.2.2 Kwiatkowski-Phillips-Schmidt-Shin (KPSS) <a name="3.2.2"></a>

# ## 3.3 The Box-Cox test <a name="3.3"></a>

# # 4. Measuring performance <a name="4"></a>

# RMSE, allong with the RMSLE, was used to compare the performance for the models. Plots were also used to show how the selected
# models perform on the data. MASE was used to calculate the accuracy of forecasts. As this uses percentages it is possible to
# compare different methods quite well.
# The three loss functions used to compare the models performance are:

# ## 4.1 Root Mean Squared Error (RMSE):

# Calculates the average error in the forecasts. Values calculated by this function represent the number of people the forecast
# got wrong.

# $ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(p_i-r_i)^2} $
# Where *p* is the predicted value, *r* is the real value and *n* is the number of data.

# ## 4.2 Root Mean Squared Log Error (RMSLE):

# Other than being chosen by Kaggle, this loss function penalises predictions smaller than the real value. So, according to this
# function, overestimating the number of infected people is better than underestimating it.

# $ RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\ln(p_i+1)-\ln(r_i+1))^2} $
# Where *p* is the predicted value, *r* is the real value and *n* is the number of data.

# ## 4.3 Mean Absolure Scaled Error (MASE):

# This represents the forecast error as a ratio. This is usefull as it allows forecasts on different scales to be compared.

# $ MASE = \frac{\frac{1}{J}\sum_{j}|p_i-r_i|}{\frac{1}{T-1}\sum_{t=2}^{T}|r_t-r_{t-1}|}  $
# Where *p* is the predicted value, *r* is the real value, *T* is the size of the training set, *J* is the number of forecats.

# # 5. Framing the problem <a name="5"></a>

# The main objective of tihs project is to use a RNN to make long range predictions on the number of confirmed, recovered and
# deceased COVID-19 cases. As the number of infected people can be inferred from the three predicted values it is not necessary
# to add it to the model thus making it simpler.

# To make these predictions for each country the model will take as input the whole time series for a country as a sample.

# The model can be expressed as a probability.
# $ P() $
# $ x^i_{t} $
# $ x^d_{t} $
# $ x^r_{t} $

# # 6. The model <a name="6"></a>

# ## Data preparation

COVID_DATA.print_n_plot_country("World")
import numpy as np
def apply_sliding_window(data, time_steps, horizon):
    """
    Implementation of the sliding window method that when applyed on multivariate time series data produces a multi step output.
    input: data (numpy array), number of time steps (int), horizon[size of output produced by time steps] (int)
    output: numpy array
    """
    x, y = [], []
    for row_idx, row in enumerate(data):
        end_x = row_idx + time_steps
        end_y = end_x + horizon
        if end_y > len(data):
            break
        sub_x, sub_y = data[row_idx:end_x], data[end_x:end_y]
        x.append(sub_x)
        y.append(sub_y)
    return np.array(x), np.array(y)

def split_train_test_set(data, test_size):
    """
    Split the data into training and testing sets for the model.
    input: data (numpy array), test_size (int)
    output: numpy array
    """
    return data[:-test_size], data[-test_size:]

def make_train_x_y(data, time_steps, horizon):
    """
    Split testing data into input and output.
    input: data (numpy array), horizon (int)
    output: x [input data] (numpy array), y [output data] (numpy array)
    """
    x, y =  [], []
    for row_idx in range(0, len(data), time_steps):
        end_x = row_idx + time_steps
        end_y = end_x + horizon

        if end_y > len(data):
            break

        x.append(data[row_idx:end_x])
        y.append(data[end_x:end_y])
    return np.array(x), np.array(y)

def plot_training_history(hist):
    fig, ax = plt.subplots()
    ax.plot(hist.history['loss'])
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['loss on train data'], loc='best')

# The healthy and infected columns were removed as they can be calculated from the data. This makes the model simpler allowing it to learn better. 
train, test = split_train_test_set(COVID_DATA.find_country("World").data.values[:,[0,1,2]], 28)

print("Tain and test")
print(train.shape)
print(test.shape)

# Standardize the data.
train_scaler = StandardScaler()
test_scaler = StandardScaler()

scaled_train = scaler.fit_transform(train)
scaled_test = scaler.fit_transform(test)

print("Scaled")
print(scaled_train.shape)
print(scaled_test.shape)

# Apply the sliding window method on the data.
train_x, train_y = apply_sliding_window(scaled_train, 7, 7)
wind_test_x, wind_test_y = apply_sliding_window(scaled_test, 7, 7)
test_x, test_y = make_train_x_y(scaled_test, 7, 7) # Make version of test data with only the three target weeks.

print("Split")
print(train_x.shape, train_y.shape)
print(wind_test_x.shape, wind_test_y.shape)
print(test_x.shape, test_y.shape)

fig, ax = plt.subplots(2,2)
ax[0,0].plot(train)
ax[0,0].set_title("Train")
ax[0,1].plot(test)
ax[0,1].set_title("Test")
ax[1,0].plot(scaled_train)
ax[1,0].set_title("Train (Scaled)")
ax[1,1].plot(scaled_test)
ax[1,1].set_title("Test (Scaled)")
fig.tight_layout()

# Comparison of original data with its starndarized version.

# ## 6.1 Multivariate Iterative Encoder Decoder LSTM

n_steps_in = 7
n_steps_out = 7
n_features = 3

ed_lstm_model = keras.Sequential()
ed_lstm_model.add(layers.LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
ed_lstm_model.add(layers.RepeatVector(n_steps_out))
ed_lstm_model.add(layers.LSTM(100, activation='relu', return_sequences=True))
ed_lstm_model.add(layers.TimeDistributed(layers.Dense(n_features)))
ed_lstm_model.compile(optimizer='adam', loss='mse')

ed_lstm_history = ed_lstm_model.fit(train_x, train_y, epochs=300, verbose=0)
ed_lstm_scores = ed_lstm_model.evaluate(test_x, test_y)
ed_lstm_wind_scores = ed_lstm_model.evaluate(wind_test_x, wind_test_y)

plot_training_history(ed_lstm_history)

ed_lstm_predictions = ed_lstm_model.predict(test_x)
ed_lstm_wind_predictions = ed_lstm_model.predict(wind_test_x)
ed_lstm_wind_predictions = np.stack([ed_lstm_wind_predictions[0], ed_lstm_wind_predictions[7], ed_lstm_wind_predictions[14]])
ed_lstm_predictions = ed_lstm_predictions.reshape(21, 3)
ed_lstm_wind_predictions = ed_lstm_wind_predictions.reshape(21, 3)

fig, ax = plt.subplots(1, 3)
ax[0].plot(scaled_test)
ax[0].set_title('Original')
ax[1].plot(ed_lstm_predictions)
ax[1].set_title('Weekly')
ax[2].plot(ed_lstm_wind_predictions)
ax[2].set_title('Windowed')
fig.tight_layout()

# The plot shows how the output of the model compares to the original data. Even though the loss for the windowed method is
# smaller than the weelky method, when ploted, the outputs look similar.

# ## 6.2 Multivariate Multi-output Iterative LSTM

def make_multi_output_data(data):
    """
    Takes a dataset with shape (sample, timestep, feature) and
    reshapes it to (feature, sample, timestep)
    """
    confirmed, deceased, recovered = [], [], []
    for sample in data:
        confirmed.append(sample[:,0])
        deceased.append(sample[:,1])
        recovered.append(sample[:,2])
    confirmed = np.stack(confirmed)
    deceased = np.stack(deceased)
    recovered = np.stack(recovered)

    return np.stack([confirmed, deceased, recovered])

print(train_y.shape)
multi_train_y = make_multi_output_data(train_y)
print(multi_train_y.shape)

print(test_y.shape)
multi_test_y = make_multi_output_data(test_y)
print(multi_test_y.shape)

print(wind_test_y.shape)
multi_wind_test_y = make_multi_output_data(wind_test_y)
print(multi_wind_test_y.shape)

inputs = keras.Input(shape=(7, 3))
hidden_lstm = layers.LSTM(100, activation='relu', return_sequences=True)(inputs)
confimed_out = layers.TimeDistributed(layers.Dense(1), name="confirmed")(hidden_lstm)
deceased_out = layers.TimeDistributed(layers.Dense(1), name="deceased")(hidden_lstm)
recovered_out = layers.TimeDistributed(layers.Dense(1), name="recovered")(hidden_lstm)

multi_o_lstm_model = keras.Model(inputs=inputs, outputs=[confimed_out, deceased_out, recovered_out])

multi_o_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])

print(multi_o_lstm_model.summary())

multi_o_lstm_history = multi_o_lstm_model.fit(train_x, [multi_train_y[0], multi_train_y[1], multi_train_y[2]], epochs=300, verbose=0)
print("Performance on weekly data.")
multi_o_lstm_scores = multi_o_lstm_model.evaluate(test_x, y=[multi_test_y[0], multi_test_y[1], multi_test_y[2]])
print("Performance on moving window data.")
multi_o_lstm_wind_scores = multi_o_lstm_model.evaluate(wind_test_x, y=[multi_wind_test_y[0], multi_wind_test_y[1], multi_wind_test_y[2]])

plot_training_history(multi_o_lstm_history)

multi_o_lstm_predictions = multi_o_lstm_model.predict(test_x)
multi_o_lstm_wind_predictions = multi_o_lstm_model.predict(wind_test_x)
multi_o_lstm_predictions = np.stack(multi_o_lstm_predictions)
multi_o_lstm_wind_predictions = np.stack(multi_o_lstm_wind_predictions)
print(multi_o_lstm_predictions.shape)
print(multi_o_lstm_wind_predictions.shape)
multi_o_lstm_predictions = multi_o_lstm_predictions.reshape(3, 21).T # Format the weekly predictions to fit the plot.
# Extract only the relevant weeks from the windowed predictions.
temp = []
for feature in multi_o_lstm_wind_predictions:
    temp.append(np.stack([feature[0], feature[7], feature[14]]))
# Now that the windowed predictions have the same shape as the weekly ones the same formatting can be applied.
multi_o_lstm_wind_predictions = np.stack(temp).reshape(3, 21).T
print(multi_o_lstm_wind_predictions.shape)

print(multi_o_lstm_predictions.shape)
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].plot(scaled_test)
ax[0].set_title('Original')
ax[0].legend(("Confirmed", "Deceased", "Recovered"))
ax[1].plot(multi_o_lstm_predictions)
ax[1].set_title('Weekly')
ax[1].legend(("Confirmed", "Deceased", "Recovered"))
ax[2].plot(multi_o_lstm_wind_predictions)
ax[2].set_title('Windowed')
ax[2].legend(("Confirmed", "Deceased", "Recovered"))
fig.tight_layout()

# ## 6.3 Multivariate Multi-output Iterative GRU

inputs = keras.Input(shape=(7, 3))
hidden_lstm = layers.GRU(100, activation='relu', return_sequences=True)(inputs)
confimed_out = layers.TimeDistributed(layers.Dense(1), name="confirmed")(hidden_lstm)
deceased_out = layers.TimeDistributed(layers.Dense(1), name="deceased")(hidden_lstm)
recovered_out = layers.TimeDistributed(layers.Dense(1), name="recovered")(hidden_lstm)

multi_o_gru_model = keras.Model(inputs=inputs, outputs=[confimed_out, deceased_out, recovered_out])

multi_o_gru_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])

print(multi_o_gru_model.summary())

multi_o_gru_history = multi_o_gru_model.fit(train_x, [multi_train_y[0], multi_train_y[1], multi_train_y[2]], epochs=300, verbose=0)
print("Performance on weekly data.")
multi_o_gru_scores = multi_o_gru_model.evaluate(test_x, y=[multi_test_y[0], multi_test_y[1], multi_test_y[2]])
print("Performance on moving window data.")
multi_o_gru_wind_scores = multi_o_gru_model.evaluate(wind_test_x, y=[multi_wind_test_y[0], multi_wind_test_y[1], multi_wind_test_y[2]])

plot_training_history(multi_o_gru_history)

multi_o_gru_predictions = multi_o_gru_model.predict(test_x)
multi_o_gru_wind_predictions = multi_o_gru_model.predict(wind_test_x)
multi_o_gru_predictions = np.stack(multi_o_gru_predictions)
multi_o_gru_wind_predictions = np.stack(multi_o_gru_wind_predictions)
print(multi_o_gru_predictions.shape)
print(multi_o_gru_wind_predictions.shape)
multi_o_gru_predictions = multi_o_gru_predictions.reshape(3, 21).T # Format the weekly predictions to fit the plot.
# Extract only the relevant weeks from the windowed predictions.
temp = []
for feature in multi_o_gru_wind_predictions:
    temp.append(np.stack([feature[0], feature[7], feature[14]]))
# Now that the windowed predictions have the same shape as the weekly ones the same formatting can be applied.
multi_o_gru_wind_predictions = np.stack(temp).reshape(3, 21).T
print(multi_o_gru_wind_predictions.shape)

print(multi_o_gru_predictions.shape)
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].plot(scaled_test)
ax[0].set_title('Original')
ax[0].legend(("Confirmed", "Deceased", "Recovered"))
ax[1].plot(multi_o_gru_predictions)
ax[1].set_title('Weekly')
ax[1].legend(("Confirmed", "Deceased", "Recovered"))
ax[2].plot(multi_o_gru_wind_predictions)
ax[2].set_title('Windowed')
ax[2].legend(("Confirmed", "Deceased", "Recovered"))
fig.tight_layout()

# # 7. Experiments <a name="7"></a>

# # 8. Analysing the results <a name="8"></a>

# | Metrics | LSTM Weekly | LSTM Windowed | GRU Weekly | GRU Windowed |
# | --- | --- | --- | --- | --- |
# | Loss | 2.2714 | 2.0155 | 1.9591 | 1.8281 |
# | Confirmed loss | 0.6823 | 0.6381 | 0.6404 | 0.6040 |
# | Deceased loss | 0.9894 | 0.7144 | 0.7427 | 0.5822 |
# | Recovered loss | 0.5997 | 0.6630 | 0.5761 | 0.6419 |
# | Confirmed MSE | 0.6823 | 0.6381 | 0.6404 | 0.6040 |
# | Confirmed RMSE | 0.8260 | 0.7988 | 0.8002 | 0.7772 |
# | Deceased MSE | 0.9894 | 0.7144 | 0.7427 | 0.5822 |
# | Deceased RMSE | 0.9947 | 0.8452 | 0.8618 | 0.7630 |
# | Recovered MSE | 0.5997 | 0.6630 | 0.5761 | 0.6419 |
# | Recovered RMSE | 0.7744 | 0.8142 | 0.7590 | 0.8012 |

# # 9. Comparison with previous experiments <a name="9"></a>

# # 10. Conclusion <a name="10"></a>
