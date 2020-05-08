import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduces tensorflow messages.

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Activation, Dropout
import talos
import data_handler

def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

get_custom_objects().update({'swish': Activation(swish)})

FORECAST_HORIZON = 4

HYPERPARAMETERS = {
        'width': [5, 10, 20, 30, 40, 50, 100],
        'activation' : ['relu', 'tanh', 'swish'],
        'dropout' : [0.1, 0.2, 0.3, 0.4, 0.5],
        'optimizer' : ['adam', 'rmsprop'],
        'loss' : ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                  'mean_squared_logarithmic_error'],
        'bach_size' : [1, 16, 32],
        'epochs' : [250, 500, 1000]
        }

def lstm_current_infected(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(LSTM(params['width'], input_shape=(x_train.shape[1], x_train.shape[2]),
                   activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['activation']))

    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    history = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['bach_size'],
                        verbose=0, validation_split=0.2)

    return history, model

def gru_current_infected(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(GRU(params['width'], input_shape=(x_train.shape[1], x_train.shape[2]),
                  activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['activation']))

    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    history = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['bach_size'],
                        verbose=0, validation_split=0.2)

    return history, model

def multivariate_lstm(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(LSTM(params['width'], return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]),
                   activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(params['width'], activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(2, activation=params['activation']))

    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    history = model.fit(x_train, y_train, epochs=400, batch_size=16, verbose=0,
                        validation_split=0.2)
    return history, model

def multivariate_gru(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(GRU(params['width'], return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]),
                  activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(GRU(params['width'], activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(2, activation=params['activation']))

    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    history = model.fit(x_train, y_train, epochs=400, batch_size=16, verbose=0,
                        validation_split=0.2)
    return history, model

def prepare_data():
    forecast_horizon = 4   # Number of observations to be used to predict the next event.
    train_set_ratio = 0.7  # The size of the training set as a percentage of the data set.

    # Setup data.
    raw_data = data_handler.load_data()
    current_infected = data_handler.calculate_current_infected(raw_data)

    # Split the data into train and test sets.
    train, test = data_handler.split_train_test(current_infected, train_set_ratio)

    # Make the time series data stationary.
    train_log, train_diff_one, train_diff_two, stationary_train = data_handler.adjust_data(train)
    test_log, test_diff_one, test_diff_two, stationary_test = data_handler.adjust_data(test)

    # Transform the data into a supervised learning dataset.
    supervised_train = data_handler.series_to_supervised(stationary_train, 0, forecast_horizon)
    supervised_test = data_handler.series_to_supervised(stationary_test, 0, forecast_horizon)

    # Create sets for input and output based on the forecast horizon.
    x_train, y_train = data_handler.split_horizon(supervised_train, forecast_horizon)
    x_test, y_test = data_handler.split_horizon(supervised_test, forecast_horizon)

    # Rescale answers to calculate the error.
    #scaled_train = data_handler.rescale_data(y_train, train_diff_one[0], train_diff_two[0], train_log[0], forecast_horizon)
    #scaled_test = data_handler.rescale_data(y_test, test_diff_one[0], test_diff_two[0], test_log[0], forecast_horizon)

    # Reshape x from [samples, time steps] to [samples, time steps, features]
    # Where samples = rows and time steps = columns.
    features = 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], features)
    return x_train, y_train

def run_talos_scan(keras_model, model_name):
    results = talos.Scan(x=X_TRAIN, y=Y_TRAIN, params=HYPERPARAMETERS, model=keras_model,
                         reduction_metric='val_loss', minimize_loss=True,
                         experiment_name=model_name, fraction_limit=0.1)
    analisys = talos.Analyze(results)
    analisys.data.to_csv(f'{model_name}.csv', index=False)
    print(f'{model_name} results')
    print(analisys.data)
    print(analisys.high('val_loss'))
    print(analisys.low('val_loss'))
    print(analisys.best_params('val_loss', [], ascending=True))

X_TRAIN, Y_TRAIN = prepare_data()
run_talos_scan(lstm_current_infected, 'lstm')
run_talos_scan(gru_current_infected, 'gru')
run_talos_scan(multivariate_lstm, 'multi_lstm')
run_talos_scan(multivariate_gru, 'multi_gru')
