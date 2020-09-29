import data
import models
import time
from tensorflow.keras.callbacks import CSVLogger, TerminateOnNaN

d = data.Data()

d.split_train_test(28)

d.standarize_data()

vocab_size, max_len, enc_names = d.encode_names()

train_x, train_y, test_x, test_y = d.retreive_data()

multi_train_y = d.make_multi_output_data(train_y)
multi_test_y = d.make_multi_output_data(test_y)

print("Data structure")
print(f"Number of countries: {len(d.countries)}")
# Check if all countries have the same data shape.
for c in d.countries:
    if c.data.shape != d.countries[0].data.shape:
        print(f"{c.name} has shape of {c.data.shape} instead of {d.countries[0].data.shape}")

print(f"Shape for country data: {d.countries[0].data.shape}")
print(f"train_x: {train_x.shape}\ntrain_y: {train_y.shape}\ntest_x: {test_x.shape}\ntest_y: {test_y.shape}")
print(f"Shape for multi train y: {multi_train_y.shape} and multi test y: {multi_test_y.shape}")
print(f"Vocab size: {vocab_size} and Max length: {max_len}")
print(f"Enc names: {enc_names.shape} and type: {type(enc_names)}")

temporal_shape = train_x[0].shape
word_shape = enc_names[0].shape
units = 100
output_size = 28

# Create models.
multi_out_lstm = models.LSTMMultiOutput(temporal_shape, word_shape, units, output_size, activation='tanh')
multi_out_gru = models.GRUMultiOutput(temporal_shape, word_shape, units, output_size)

multi_out_lstm_V2 = models.LSTMMultiOutput_V2(temporal_shape, word_shape, units, output_size, activation='tanh')
multi_out_gru_V2 = models.GRUMultiOutput_V2(temporal_shape, word_shape, units, output_size)

single_out_lstm = models.LSTMSingleOutput(temporal_shape, word_shape, units, output_size, activation='tanh')
single_out_gru = models.GRUSingleOutput(temporal_shape, word_shape, units, output_size)

multi_lstm_quant = models.LSTMMultiOutputQuantile(temporal_shape, word_shape, units, output_size, activation='tanh')
multi_gru_quant = models.GRUMultiOutputQuantile(temporal_shape, word_shape, units, output_size)

single_lstm_quant = models.LSTMSingleOutputQuantile(temporal_shape, word_shape, units, output_size, activation='tanh')
single_gru_quant = models.GRUSingleOutputQuantile(temporal_shape, word_shape, units, output_size)

# Print model architecture.
print(multi_out_lstm.summary())
print(multi_out_gru.summary())
print(multi_out_lstm_V2.summary())
print(multi_out_gru_V2.summary())
print(single_out_lstm.summary())
print(single_out_gru.summary())
print(multi_lstm_quant.summary())
print(multi_gru_quant.summary())
print(single_lstm_quant.summary())
print(single_gru_quant.summary())

# Create logger callbacks.
multi_out_lstm_logger = CSVLogger('multi_out_lstm.csv', separator=',')
multi_out_gru_logger = CSVLogger('multi_out_gru.csv', separator=',')
multi_out_lstm_V2_logger = CSVLogger('multi_out_lstm_V2.csv', separator=',')
multi_out_gru_V2_logger = CSVLogger('multi_out_gru_V2.csv', separator=',')
single_out_lstm_logger = CSVLogger('single_out_lstm.csv', separator=',')
single_out_gru_logger = CSVLogger('single_out_gru.csv', separator=',')

ton_back = TerminateOnNaN()

epochs = 300
verbose=1

# Train models.
multi_out_lstm_hist = multi_out_lstm.fit([train_x, enc_names], [multi_train_y[0], multi_train_y[1], multi_train_y[2]],
                                         epochs=epochs, verbose=verbose, callbacks=[multi_out_lstm_logger, ton_back])

multi_out_gru_hist = multi_out_gru.fit([train_x, enc_names], [multi_train_y[0], multi_train_y[1], multi_train_y[2]],
                                       epochs=epochs, verbose=verbose, callbacks=[multi_out_gru_logger, ton_back])

multi_out_lstm_V2_hist = multi_out_lstm_V2.fit([train_x, enc_names], [multi_train_y[0], multi_train_y[1], multi_train_y[2]],
                                               epochs=epochs, verbose=verbose, callbacks=[multi_out_lstm_V2_logger,
                                                                                          ton_back])

multi_out_gru_V2_hist = multi_out_gru_V2.fit([train_x, enc_names], [multi_train_y[0], multi_train_y[1], multi_train_y[2]],
                                             epochs=epochs, verbose=verbose, callbacks=[multi_out_gru_V2_logger, ton_back])

single_out_lstm_hist = single_out_lstm.fit([train_x, enc_names], train_y,
                                           epochs=epochs, verbose=verbose, callbacks=[single_out_lstm_logger, ton_back])

single_out_gru_hist = single_out_gru.fit([train_x, enc_names], train_y,
                                         epochs=epochs, verbose=verbose, callbacks=[single_out_gru_logger, ton_back])

y_multi = {"confirmed_q1": multi_train_y[0],
     "confirmed_q2": multi_train_y[0],
     "confirmed_q3": multi_train_y[0],
     "deceased_q1": multi_train_y[1],
     "deceased_q2": multi_train_y[1],
     "deceased_q3": multi_train_y[1],
     "recovered_q1": multi_train_y[2],
     "recovered_q2": multi_train_y[2],
     "recovered_q3": multi_train_y[2]}


multi_lstm_hist = multi_lstm_quant.fit([train_x, enc_names], y=y_multi, epochs=epochs, verbose=verbose, callbacks=[ton_back])

multi_gru_hist = multi_gru_quant.fit([train_x, enc_names], y=y_multi, epochs=epochs, verbose=verbose, callbacks=[ton_back])

y_single = {"output_q1": train_y,
            "output_q2": train_y,
            "output_q3": train_y}

single_lstm_hist = single_lstm_quant.fit([train_x, enc_names], y=y_single, epochs=epochs, verbose=verbose, callbacks=[ton_back])

single_gru_hist = single_gru_quant.fit([train_x, enc_names], y=y_single, epochs=epochs, verbose=verbose, callbacks=[ton_back])

# Evaluate models.
multi_out_lstm_eval = multi_out_lstm.evaluate([test_x, enc_names], [multi_test_y[0], multi_test_y[1], multi_test_y[2]],
                                              return_dict=True)

multi_out_gru_eval = multi_out_gru.evaluate([test_x, enc_names], [multi_test_y[0], multi_test_y[1], multi_test_y[2]],
                                            return_dict=True)

multi_out_lstm_V2_eval = multi_out_lstm_V2.evaluate([test_x, enc_names], [multi_test_y[0], multi_test_y[1], multi_test_y[2]],
                                                    return_dict=True)

multi_out_gru_V2_eval = multi_out_gru_V2.evaluate([test_x, enc_names], [multi_test_y[0], multi_test_y[1], multi_test_y[2]],
                                                  return_dict=True)

single_out_lstm_eval = single_out_lstm.evaluate([test_x, enc_names], test_y, return_dict=True)

single_out_gru_eval = single_out_gru.evaluate([test_x, enc_names], test_y, return_dict=True)

multi_lstm_eval = multi_lstm_quant.evaluate([test_x, enc_names], [multi_train_y[0], multi_train_y[0], multi_train_y[0],
                                                                  multi_train_y[1], multi_train_y[1], multi_train_y[1],
                                                                  multi_train_y[2], multi_train_y[2], multi_train_y[2]],
                                            return_dict=True)

multi_gru_eval = multi_gru_quant.evaluate([test_x, enc_names], [multi_train_y[0], multi_train_y[0], multi_train_y[0],
                                                                multi_train_y[1], multi_train_y[1], multi_train_y[1],
                                                                multi_train_y[2], multi_train_y[2], multi_train_y[2]],
                                          return_dict=True)

single_lstm_eval = single_lstm_quant.evaluate([test_x, enc_names], [test_y, test_y, test_y], return_dict=True)

single_gru_eval = single_gru_quant.evaluate([test_x, enc_names], [test_y, test_y, test_y], return_dict=True)

# Does multi output affect performance? In theory the models should be the same.
# Shared parameters use the TimeDistributed function while using the Dense size will create individual parameters for each day.
# To have the model output three quantiles use multi output with each output node optimized on a pinball loss with a different
# quantile value.

# Callbacks: TerminateOnNAN
# Models (one LSTM and one GRU):
# Multi output individual. (OK)
# Multi output shared. (OK)
# Single output shared. (OK)
# Multi output quantile.

# Test tanh on multi output LSTM models.
# Test regularization L1 and/or L2.
# Apply gradient clipping.
import matplotlib.pyplot as plt

def plot_training_history(hist, title):
    fig, ax = plt.subplots()
    ax.plot(hist.history['loss'])
    ax.plot(hist.history['confirmed_loss'])
    ax.plot(hist.history['deceased_loss'])
    ax.plot(hist.history['recovered_loss'])
    ax.set_title(f'model loss on train data: {title}')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['loss', 'confirmed_loss', 'deceased_loss', 'recovered_loss'], loc='best')

def plot_training_history_single(hist, title):
    fig, ax = plt.subplots()
    ax.plot(hist.history['loss'])
    ax.set_title(f'model loss on train data: {title}')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['loss'], loc='best')

def plot_training_metrics(hist, title):
    fig, ax = plt.subplots()
    ax.plot(hist.history['confirmed_mean_squared_error'])
    ax.plot(hist.history['confirmed_root_mean_squared_error'])
    ax.plot(hist.history['deceased_mean_squared_error'])
    ax.plot(hist.history['deceased_root_mean_squared_error'])
    ax.plot(hist.history['recovered_mean_squared_error'])
    ax.plot(hist.history['recovered_root_mean_squared_error'])
    ax.set_title(f'model error metrics on train data: {title}')
    ax.set_ylabel('error')
    ax.set_xlabel('epoch')
    ax.legend(['confirmed_mean_squared_error', 'confirmed_root_mean_squared_error', 'deceased_mean_squared_error', 'deceased_root_mean_squared_error', 'recovered_mean_squared_error', 'recovered_root_mean_squared_error'], loc='best')

def plot_training_metrics_single(hist, title):
    fig, ax = plt.subplots()
    ax.plot(hist.history['root_mean_squared_error'])
    ax.plot(hist.history['mean_squared_error'])
    ax.set_title(f'model error metrics on train data: {title}')
    ax.set_ylabel('error')
    ax.set_xlabel('epoch')
    ax.legend(['root_mean_squared_error', 'mean_squared_error'], loc='best')

plot_training_history(multi_out_lstm_hist, "multi_out_lstm_hist")
plot_training_history(multi_out_gru_hist, "multi_out_gru_hist")
plot_training_history(multi_out_lstm_V2_hist, "multi_out_lstm_V2_hist")
plot_training_history(multi_out_gru_V2_hist, "multi_out_gru_V2_hist")
plot_training_history_single(single_out_lstm_hist, "single_out_lstm_hist")
plot_training_history_single(single_out_gru_hist, "single_out_gru_hist")

plot_training_metrics(multi_out_lstm_hist, "multi_out_lstm_hist")
plot_training_metrics(multi_out_gru_hist, "multi_out_gru_hist")
plot_training_metrics(multi_out_lstm_V2_hist, "multi_out_lstm_V2_hist")
plot_training_metrics(multi_out_gru_V2_hist, "multi_out_gru_V2_hist")
plot_training_metrics_single(single_out_lstm_hist, "single_out_lstm_hist")
plot_training_metrics_single(single_out_gru_hist, "single_out_gru_hist")

from pprint import pprint
print("multi_out_lstm_eval")
pprint(multi_out_lstm_eval)
print()
print("multi_out_gru_eval")
pprint(multi_out_gru_eval)
print()
print("multi_out_lstm_V2_eval")
pprint(multi_out_lstm_V2_eval)
print()
print("multi_out_gru_V2_eval")
pprint(multi_out_gru_V2_eval)
print()
print("single_out_lstm_eval")
pprint(single_out_lstm_eval)
print()
print("single_out_gru_eval")
pprint(single_out_gru_eval)
