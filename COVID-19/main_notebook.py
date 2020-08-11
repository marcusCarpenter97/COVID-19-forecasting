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
import data
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
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

# # 3. Time series <a name="3"></a>

# ## 3.1 Autocorrelation <a name="3.1"></a>

# When making predictions based on a time series it is important to check whether the data is random or not. If the time series
# happens to be random then no model would be able to make a accurate prediction. This can be tested with an autocorrelation
# plot. Essencialy this compares how similar the data is to it self by shifting it by a certain amount. These shifts are called
# lags. When the autocorrelation plot shows a value of 1 the time series is identical to its lagged version, if the plot shows a
# 0 then the values are random. Any values outside of the shaded area are considered to be statistically significant.

fig, ax = plt.subplots(3, 2, figsize=(15, 10))
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Confirmed"], title="Confirmed", ax=ax[0,0])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Deceased"], title="Deceased", ax=ax[0,1])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Recovered"], title="Recovered", ax=ax[1,0])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Infected"], title="Infected", ax=ax[1,1])
temp_ax = plot_acf(COVID_DATA.find_country("World").data["Healthy"], title="Healthy", ax=ax[2,0])
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
# perturbation mmaking the values on the y-axis go up or down, however, the mean will always follow the trend. An example of
# this is

# ### 3.2.1 Augmented Dickey Fuller (ADF) <a name="3.2.1"></a>

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
# # 7. Experiments <a name="7"></a>
# # 8. Analysing the results <a name="8"></a>
# # 9. Comparrison with previous experiments <a name="9"></a>
# # 10. Conclusion <a name="10"></a>
