import pandas as pd
import data_handler

class Data:
    """ A container for what data_handler produces."""
    def __init__(self):
        self.confirmed, self.deceased, self.recovered = data_handler.load_data()

    def calculate_current_infected(self):
        self.current_infected = data_handler.calculate_current_infected((self.confirmed,
                                                                         self.deceased,
                                                                         self.recovered))

    def make_kaggle_data(self):
        self.confirmed = self.confirmed.sum()[2:]
        self.deceased = self.deceased.sum()[2:]
        self.kaggle_data = pd.concat([self.confirmed, self.deceased], axis=1)

    def split_train_test(self, data):
        self.train, self.test = data_handler.split_train_test(data)

    def prepare_data(self):
        self.train_log, self.train_diff_one, self.train_diff_two, self.stationary_train = data_handler.adjust_data(self.train)
        self.test_log, self.test_diff_one, self.test_diff_two, self.stationary_test = data_handler.adjust_data(self.test)

    def rescale_data(self, train_data, test_data, forecast_horizon):
        scaled_train = data_handler.rescale_data(train_data, self.train_diff_one[0],
                                                             self.train_diff_two[0],
                                                             self.train_log[0],
                                                             forecast_horizon)
        scaled_test = data_handler.rescale_data(test_data, self.test_diff_one[0],
                                                           self.test_diff_two[0],
                                                           self.test_log[0],
                                                           forecast_horizon)
        return scaled_train, scaled_test

    def series_to_supervised(self, forward_offset, backward_offset=0, multivariate=False):
        if multivariate:
            self.x_train, self.y_train = data_handler.multivariate_to_supervised(self.stationary_train, forward_offset)
            self.x_test, self.y_test = data_handler.multivariate_to_supervised(self.stationary_test, forward_offset)
        else:
            self.supervised_train = data_handler.series_to_supervised(self.stationary_train, backward_offset, forward_offset)
            self.supervised_test = data_handler.series_to_supervised(self.stationary_test, backward_offset, forward_offset)

    def split_in_from_out(self, offset, features):
        self.x_train, self.y_train = data_handler.split_horizon(self.supervised_train, offset)
        self.x_test, self.y_test = data_handler.split_horizon(self.supervised_test, offset)

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], features)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], features)
