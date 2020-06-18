from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import math
import numpy as np
import pandas as pd
import data_loader
import country

class Data:
    """ A container for what the data_loader loads."""
    def __init__(self):
        self.raw_confirmed, self.raw_deceased, self.raw_recovered = data_loader.load_covid_data()
        self.population = data_loader.load_population_data()
        self.countries = []

        def generate_world_data():
            p = self.population.sum(numeric_only=True)['Population']
            c = self.raw_confirmed.sum(numeric_only=True)
            d = self.raw_deceased.sum(numeric_only=True)
            r = self.raw_recovered.sum(numeric_only=True)
            i = self.calculate_current_infected(c, d, r)
            h = self.calculate_healthy(p, d, r, i)
            return country.Country("World", p, c, d, r, i, h)

        def generate_countries():
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
        Takes a country's name and returns its object object.
        Returns None if country is not found.
        """
        res = [country for country in self.countries if country.name == name]
        return None if len(res) == 0 else res[0]

    def encode_names(self, extra_size=1.25):
        """
        Creates encoded versions of the country names.
        These will be used in the Embedding layer of the network.
        Param: extra_size float - extra space to guarantee uniqueness.
        Returns:
        vocab_size - int - number of unique words plus some extra space
        max_length - int - size of biggest word.
        """
        # Idealy this should use the number of words not the country names.
        vocab_size = math.ceil(len(self.countries) * extra_size)

        encoded = [one_hot(country.name, vocab_size) for country in self.countries]
        max_length = len(max(encoded, key=lambda x: len(x)))

        padded = pad_sequences(encoded, maxlen=max_length, padding='post')

        for country, enc_name in zip(self.countries, padded):
            country.encoded_name = enc_name

        return vocab_size, max_length

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
        for country in self.countries:
            country.log_data()

    def exp(self):
        """
        Apply the exp function on all countries.
        """
        for country in self.countries:
            country.exp_data()

    def difference(self):
        """
        Apply the diff function on all countries.
        """
        for country in self.countries:
            country.diff_data()

    def integrate(self):
        """
        Integrate the data for all countries.
        """
        for country in self.countries:
            country.int_data()

    def split_train_test(self, test_size, horizon):
        """
        Create training and testing datasets for all countries.
        """
        for country in self.countries:
            country.split_data(test_size, horizon)

    def supervise_data(self, horizon):
        """
        Create input and output test sets for all countries.
        """
        for country in self.countries:
            country.supervise_data(horizon)

    # TODO unsused.
    def get_ts_samples(self, start, end):
        return np.array([np.array(country.get_slice(start, end)) for country in self.countries])

    # TODO unsused.
    def get_encoded_names(self):
        names = np.array([country.encoded_name for country in self.countries])
        return names.reshape(names.shape[0], 1, names.shape[1])
