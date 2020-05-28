import pandas as pd
import data_handler
import country

class Data:
    """ A container for what data_handler produces."""
    def __init__(self):
        self.raw_confirmed, self.raw_deceased, self.raw_recovered = data_handler.load_covid_data()
        self.population = data_handler.load_population_data()

        def generate_global_data():
            p = self.population.sum(numeric_only=True)['Population']
            c = self.raw_confirmed.sum(numeric_only=True)
            d = self.raw_deceased.sum(numeric_only=True)
            r = self.raw_recovered.sum(numeric_only=True)
            i = self.calculate_current_infected(c, d, r)
            h = self.calculate_healthy(p, d, r, i)
            return pd.DataFrame([c, d, r, i, h], columns=["Confirmed", "Deceased", "Recovered", "Infected", "Healthy"])

        def generate_country_data():
            countries = []
            # For each country in population.
            for name, pop in self.population.iterrows():
                p = pop['Population']
                # Get all relevant time series based on country name.
                c = self.raw_confirmed.loc[self.raw_confirmed['Country/Region'] == name].sum(numeric_only=True)
                d = self.raw_deceased.loc[self.raw_deceased['Country/Region'] == name].sum(numeric_only=True)
                r = self.raw_recovered.loc[self.raw_recovered['Country/Region'] == name].sum(numeric_only=True)
                i = self.calculate_current_infected(c, d, r)
                h = self.calculate_healthy(p, d, r, i)
                # Create new country object.
                countries.append(country.Country(name, p, c, d, r, i, h))
            return countries

        self.global_data = generate_global_data()
        self.country_data = generate_country_data()

    def find_country(self, name):
        """
        Takes a country's name and returns its object object.
        Returns None if country is not found.
        """
        res = [country for country in self.country_data if country.name == name]
        return None if len(res) == 0 else res[0]

    def calculate_current_infected(self, c, d, r):
        """
        Infected people = confirmed - (dead + recovered)
        """
        return c - (d + r)

    def calculate_healthy(self, p, d, r, i):
        """
        Healthy people = population - (dead + recovered + infected)
        """
        return p - (d + r + i)

    def log(self):
        """
        Apply the log function on all countries.
        """
        for country in self.country_data:
            country.log_data()

    def exp(self):
        """
        Apply the exp function on all countries.
        """
        for country in self.country_data:
            country.exp_data()

    def difference(self):
        """
        Apply the diff function on all countries.
        """
        for country in self.country_data:
            country.diff_data()

    def integrate(self):
        """
        Integrate the data for all countries.
        """
        for country in self.country_data:
            country.int_data()

    def print_global(self):
        print(self.global_data)

    def plot_global(self):
        self.global_data[["Confirmed", "Deceased", "Recovered", "Infected"]].plot()

# TODO remove all below.
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
