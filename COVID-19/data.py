import numpy as np
import pandas as pd
import hashlib
import data_loader
import country

class Data:
    """ A container that stores the time series COVID-19 data for each country.
        The data is loaded from the data_loader module.
    """
    def __init__(self):
        self.raw_confirmed, self.raw_deceased, self.raw_recovered = data_loader.load_covid_data()
        self.population = data_loader.load_population_data()
        self.countries = []

        def generate_world_data():
            """
                Creates the World data by summing all the individual
                time series data for all the countries in the dataset.

                Returns:
                    A Country object containing the world data.
            """
            p = self.population.sum(numeric_only=True)['Population']
            c = self.raw_confirmed.sum(numeric_only=True)
            d = self.raw_deceased.sum(numeric_only=True)
            r = self.raw_recovered.sum(numeric_only=True)
            i = self.calculate_current_infected(c, d, r)
            h = self.calculate_healthy(p, d, r, i)
            return country.Country("World", p, c, d, r, i, h)

        def generate_countries():
            """
                Creates a list of Country objects each containing the COVID-19 data for their respective country.
                Retunrs:
                    A list of Country objects.
            """
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

        self.countries.append(generate_world_data())
        self.countries.extend(generate_countries())

    def find_country(self, name):
        """
        Takes a country's name and returns its country object.
        Returns None if country is not found.
        """
        res = [country for country in self.countries if country.name == name]
        return None if len(res) == 0 else res[0]

    def encode_names(self, output_space=6):
        """
        Creates hashed versions of the country names.
        These will be used in the Embedding layer of the network.
        Param: output_space - int - size of the poutput produced by the hash functions to guarantee uniqueness.
        Returns:
        hashed_names - list - a list containing all the hashed names for the countries.
        """
        hashed_names = [int(hashlib.sha256(country.encode('utf-8')).hexdigest(), 16) % 10**output_space for country in
                        self.population]

        for country, hashed_name in zip(self.countries, hashed_names):
            country.encoded_name = hashed_name

        return hashed_names

    def calculate_current_infected(self, c, d, r):
        """
        Infected people = confirmed - (dead + recovered)
        Params: all Series objects.
        Returns: Series.
        """
        return c - (d + r)

    def calculate_healthy(self, p, d, r, i):
        """
        Healthy people = population - (dead + recovered + infected)
        Params: all Series objects.
        Returns: Series.
        """
        return p - (d + r + i)

    # TODO ?
    def log(self):
        """
        Apply the log function on all countries.
        """
        for country in self.countries:
            country.log_data()

    # TODO ?
    def exp(self):
        """
        Apply the exp function on all countries.
        """
        for country in self.countries:
            country.exp_data()

    # TODO ?
    def difference(self):
        """
        Apply the diff function on all countries.
        """
        for country in self.countries:
            country.diff_data()

    # TODO ?
    def integrate(self):
        """
        Integrate the data for all countries.
        """
        for country in self.countries:
            country.int_data()

    def split_train_test(self, test_size):
        """
        Create training and testing datasets for all countries.
        """
        for country in self.countries:
            country.split_data(test_size)

    def standarize_data(self):
        """
        Standarize the train and test data for all countries.
        """
        for country in self.countries:
            country.standarize()

    def destandarize_data(self, predictions):
        """
        Destandarize the predictions for all countries.
        """
        return np.stack([country.destandarize(prediction) for country, prediction in zip(self.countries, predictions)])

    def retreive_data(self):
        """
        Compile the datasets used by a machine learning model from the data of all countries.
        """
        train_x, train_y, test_x, test_y = [], [], [], []
        for country in self.countries:
            train_x.append(country.scaled_train_x)
            train_y.append(country.scaled_train_y)
            test_x.append(country.scaled_test_x)
            test_y.append(country.scaled_test_y)
        return np.stack(train_x), np.stack(train_y), np.stack(test_x), np.stack(test_y)

    def make_multi_output_data(self, data):
        """
        Takes a dataset with shape (sample, timestep, feature) and
        reshapes it to (feature, sample, timestep)
        Returns: 3D numpy array
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

    # TODO ?
    def apply_sliding_window(self, time_steps, horizon):
        """
        Apply the sliding window to preprocess the data for all countries.
        """
        for country in self.countries:
            country.apply_sliding_window(country.train, time_steps, horizon)
            country.apply_sliding_window(country.test, time_steps, horizon, is_test_data=True)

    # TODO !
    def supervise_data(self, horizon):
        """
        Create input and output test sets for all countries.
        """
        for country in self.countries:
            country.supervise_data(horizon)

    def print_n_plot_country(self, name, bars=[]):
        """
        Display a country's data.

        Parametes
        country: str - a countries name.
        bars: list - list of x coordinates
        defining where to place vertical bars.
        """
        country = self.find_country(name)

        country.print_country()

        if len(bars) > 0:
            country.plot_country(train_date=bars)
        else:
            country.plot_country()

    # TODO unsused.
    def get_ts_samples(self, start, end):
        return np.array([np.array(country.get_slice(start, end)) for country in self.countries])

    # TODO unsused.
    def get_encoded_names(self):
        names = np.array([country.encoded_name for country in self.countries])
        return names.reshape(names.shape[0], 1, names.shape[1])
