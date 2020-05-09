import numpy as np
#import pandas as pd
import talos
import talos_test
import data

FORECAST_HORIZON = 4   # Number of observations to be used to predict the next event.
UNIVARIATE = 1
MULTIVARIATE = 2

# Loss functions to evalues performance.
def rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())

def rmsle(prediction, target):
    return np.sqrt(((np.log(prediction+1) - np.log(target+1)) ** 2).mean())

def mase(prediction, target, train_size):
    mean_error = np.mean(np.abs(prediction - target))
    scaling_factor = (1/(train_size-1)) * np.sum(np.abs(np.diff(target.ravel())))
    return mean_error / scaling_factor

def weighted_pinball_loss(prediction, target, quantiles, fatal_weight=False):
    # Let the weight be log(country population +1)^-1 if error is for confirmed cases
    #                   10 * log(country population +1)^-1 if error is for fatalities
    # For each forecast: 
    #  For each quantile:
    #   Calculate the Pinball Loss.
    #  Let quntile_sum be the sum all losses for all quantiles.
    #  Let quantile_ration be 1 divided by the total number of quantiles.
    #  Let weighted_loss be the product of the weight, the quantile_ratio and the quantile_sum.
    # Let weighted_loss_sum be the sum of all weighted_loss values.
    # Let forecast_ratio be 1 divided by the total number of forecasts.
    # Let score be the product of forecast_ratio and the weighted_loss_sum.
    pass

# Univariate data
uni_covid_data = data.Data()
uni_covid_data.calculate_current_infected()
uni_covid_data.split_train_test(uni_covid_data.current_infected)
uni_covid_data.prepare_data()
uni_covid_data.series_to_supervised(FORECAST_HORIZON)
uni_covid_data.split_in_from_out(FORECAST_HORIZON, UNIVARIATE)
UNI_Y_TRAIN, UNI_Y_TEST = uni_covid_data.rescale_data(uni_covid_data.y_train, uni_covid_data.y_test, FORECAST_HORIZON)
# Multivariate data
multi_covid_data = data.Data()
multi_covid_data.make_kaggle_data()
multi_covid_data.split_train_test(multi_covid_data.kaggle_data)
multi_covid_data.prepare_data()
multi_covid_data.series_to_supervised(FORECAST_HORIZON, multivariate=True)
MULTI_Y_TRAIN, MULTI_Y_TEST = multi_covid_data.rescale_data(multi_covid_data.y_train, multi_covid_data.y_test, FORECAST_HORIZON)

# Load talos results.
lstm_res = talos.Analyze('lstm.csv')
gru_res = talos.Analyze('gru.csv')
multi_lstm_res = talos.Analyze('multi_lstm.csv')
multi_gru_res = talos.Analyze('multi_gru.csv')

# Get the top five results.
top_lstm = lstm_res.table('val_loss', sort_by='val_loss', ascending=True)[:5]
top_gru = gru_res.table('val_loss', sort_by='val_loss', ascending=True)[:5]
top_multi_lstm = multi_lstm_res.table('val_loss', sort_by='val_loss', ascending=True)[:5]
top_multi_gru = multi_gru_res.table('val_loss', sort_by='val_loss', ascending=True)[:5]

top_lstm.rename(columns={"loss":"loss_train", "loss.1": "loss" }, inplace=True)
top_gru.rename(columns={"loss":"loss_train", "loss.1": "loss" }, inplace=True)
top_multi_lstm.rename(columns={"loss":"loss_train", "loss.1": "loss" }, inplace=True)
top_multi_gru.rename(columns={"loss":"loss_train", "loss.1": "loss" }, inplace=True)

print('LSTM results')
print(top_lstm)
print('GRU results')
print(top_gru)
print('Multi LSTM results')
print(top_multi_lstm)
print('Multi GRU results')
print(top_multi_gru)

def test_top_res(covid_data, model_name, top_res, model, scaled_train, scaled_test):
    for idx, row in top_res.iterrows():
        print(idx)
        rmse_train = []
        rmse_test = []
        rmsle_train = []
        rmsle_test = []
        mase_train = []
        mase_test = []
        for i in range(10):  # Number of tests was chosen arbitrarily.
            print(i)
            # Create and train the model.
            hist, trained_model = model(covid_data.x_train, covid_data.y_train, [], [], row)
            # Make predictions.
            train_pred = trained_model.predict(covid_data.x_train)
            test_pred = trained_model.predict(covid_data.x_test)
            # Rescale predictions to original scale.
            scaled_train_pred, scaled_test_pred = covid_data.rescale_data(train_pred, test_pred, FORECAST_HORIZON)

            # Calculate error values for train and test data.
            rmse_train.append(rmse(scaled_train_pred, scaled_train))
            rmse_test.append(rmse(scaled_test_pred, scaled_test))

            rmsle_train.append(rmsle(scaled_train_pred, scaled_train))
            rmsle_test.append(rmsle(scaled_test_pred, scaled_test))

            mase_train.append(mase(scaled_train_pred, scaled_train, len(covid_data.x_train)))
            mase_test.append(mase(scaled_test_pred, scaled_test, len(covid_data.x_train)))

        # Generate performance report for model.
        report = (f"\n\nModel name: {idx}\n RMSE on train: {rmse_train}\n RMSE on test: {rmse_test}\n Average RMSE: "
        f"{np.array(rmse_test).mean()}\n RMSLE on train: {rmsle_train}\n RMSLE on test: {rmsle_test}\n Average RMSLE: "
        f"{np.array(rmsle_test).mean()}\n MASE on train: {mase_train}\n MASE on test: {mase_test}\n Average MASE: "
        f"{np.array(mase_test).mean()}\n Hyperparameters used:\n{row}\n")

        with open(f"{model_name}.txt", 'a') as res_file:
            res_file.write(report)

test_top_res(uni_covid_data, "lstm", top_lstm, talos_test.lstm_current_infected, UNI_Y_TRAIN, UNI_Y_TEST)
test_top_res(uni_covid_data, "gru", top_gru, talos_test.gru_current_infected, UNI_Y_TRAIN, UNI_Y_TEST)
test_top_res(multi_covid_data, "multi_lstm", top_multi_lstm, talos_test.multivariate_lstm, MULTI_Y_TRAIN, MULTI_Y_TEST)
test_top_res(multi_covid_data, "multi_gru", top_multi_gru, talos_test.multivariate_gru, MULTI_Y_TRAIN, MULTI_Y_TEST)
