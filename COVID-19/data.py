import pandas as pd
import data_handler
import country

class Data:
    """ A container for what data_handler produces."""
    def __init__(self):
        self.raw_confirmed, self.raw_deceased, self.raw_recovered = data_handler.load_covid_data()
        self.global_confirmed = self.raw_confirmed.sum(numeric_only=True)
        self.global_deceased = self.raw_deceased.sum(numeric_only=True)
        self.global_recovered = self.raw_recovered.sum(numeric_only=True)

        def generate_country_data():
            population = data_handler.load_population_data()
            countries = []
            # For each country in population.
            for name, pop in population.iterrows():
                # Get all relevant time series based on counrty name.
                c = self.raw_confirmed.loc[self.raw_confirmed['Country/Region'] == name]
                d = self.raw_deceased.loc[self.raw_deceased['Country/Region'] == name]
                r = self.raw_recovered.loc[self.raw_recovered['Country/Region'] == name]
                # Create new country object.
                countries.append(country.Country(name, pop, c, d, r))
            return countries
        self.country_data = generate_country_data()

    def find_country(self, name):
        """
        Takes a country's name and returns its object object.
        Returns None if country is not found.
        """
        res = [country for country in self.country_data if country.name == name]
        return None if len(res) == 0 else res[0]

    def calculate_current_infected(self, country_name=None):
        """
        Infected people = confirmed - (dead + recovered)
        If no country is specified it defaults to the global data.
        """
        if country_name:
            country = self.find_country(country_name)
            if country:
                country.infected = country.confirmed - (country.deceased + country.recovered)
        else:
            self.global_infected = self.global_confirmed - (self.global_deceased + self.global_recovered)

    def calculate_healthy(self):
        self.global_healthy = self.global_population - (self.deceased + self.recovered + self.global_infected)

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
