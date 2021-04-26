import hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
import data_loader
import country

class Data:
    """ A container that stores the time series COVID-19 data for each country.
        The data is loaded from the data_loader module.
    """
    def __init__(self):
        self.loader = data_loader.DataLoader()
        self.raw_confirmed, self.raw_deceased, self.raw_recovered = self.loader.load_covid_data()
        self.population = self.loader.load_population_data()
        self.countries = []
        self.pad_val = -10000
        self.horizon = 28
        self.features = 3
        self.padding = np.full((self.horizon, self.features), self.pad_val)

    def prepare_data(self):
        self.populate_countries()
        self.encode_names()
        self.split_data()

        scaled_train, _ = self.standardize(self.train)
        scaled_val, self.val_scalers = self.standardize(self.val)
        scaled_test_x, _ = self.standardize(self.test_x)
        scaled_test_y, self.test_y_scalers = self.standardize(self.test_y)

        self.padded_scaled_train = self.pad_data(scaled_train)
        self.padded_scaled_test_x = self.pad_data(scaled_test_x, offset=1)

        self.multi_out_scaled_val = self.prepare_output_data(scaled_val)
        self.multi_out_scaled_test_y = self.prepare_output_data(scaled_test_y)

    def populate_countries(self):
        """
            Creates a list of Country objects each containing the COVID-19 data for their respective country.
            Retunrs:
                A list of Country objects.
        """
        # For each country in population.
        for name, pop in self.population.iterrows():
            p = pop['Population']
            # Get all relevant time series based on country name.
            c = self.raw_confirmed.loc[self.raw_confirmed['Country/Region'] == name].sum(numeric_only=True)
            d = self.raw_deceased.loc[self.raw_deceased['Country/Region'] == name].sum(numeric_only=True)
            r = self.raw_recovered.loc[self.raw_recovered['Country/Region'] == name].sum(numeric_only=True)
            # Create new country object.
            self.countries.append(country.Country(name, p, c, d, r))

    def encode_names(self, output_space=6):
        """
        Creates hashed versions of the country names.
        These will be used in the Embedding layer of the network.
        Param: output_space - int - size of the poutput produced by the hash functions to guarantee uniqueness.
        Returns:
        hashed_names - list - a list containing all the hashed names for the countries.
        """
        # Hash all country names using SHA-256 then convert the hex output to int slice of the first digits by converting it to
        # a string and save it as an int in a list.
        hashed_names = [int(str(int(hashlib.sha256(country.encode('utf-8')).hexdigest(), 16))[:output_space]) for country in
                        self.population.index]

        # Convert the integers into an array of digits.
        hashed_names = np.stack([np.array(list(map(int,str(x)))) for x in hashed_names])

        for country, hashed_name in zip(self.countries, hashed_names):
            country.encoded_name = hashed_name

        return hashed_names

    def cross_validate(self, train_size):
        """
        Cross validate data for all countries.
        """
        train, val, test_x, test_y = [], [], [], []
        for country in self.countries:
            tr, v, te_x, te_y = country.split_k_fold(train_size, self.horizon)
            train.append(tr), val.append(v), test_x.append(te_x), test_y.append(te_y)
        return np.stack(train), np.stack(val), np.stack(test_x), np.stack(test_y)

    def split_data(self):
        """
        Returns four lists of numpy arrrays. Each list has the shape (fold, countries, days). The number of days changes over the
        folds.
        """
        self.train, self.val, self.test_x, self.test_y = [], [], [], []
        train_size = self.horizon
        # This assumes all countries have the same length.
        # The minus two gives space for the validation and test sets as they will overshoot.
        k_folds = len(self.countries[0].data)//self.horizon - 2
        for _ in range(k_folds):
            tr, v, te_x, te_y = self.cross_validate(train_size)
            self.train.append(tr), self.val.append(v), self.test_x.append(te_x), self.test_y.append(te_y)
            train_size += self.horizon

    def standardize(self, data):
        scaled_data = []
        scalers = []
        for fold in data:
            scaled_fold = []
            scalers_fold = []
            for country in fold:
                scaler = StandardScaler()
                scaled_fold.append(scaler.fit_transform(country))
                scalers_fold.append(scaler)
            scaled_data.append(np.stack(scaled_fold))
            scalers.append(scalers_fold)
        return scaled_data, scalers

    def pad_data(self, data, offset=0):
        padded_scaled_data = []
        folds = len(data) - offset
        for fold in data:
            padded_fold = []
            fold_padding = np.repeat(self.padding, folds).reshape(self.horizon*folds, self.features)
            for row in fold:
                padded_fold.append(np.append(row, fold_padding, axis=0))
            folds -= 1
            padded_scaled_data.append(np.stack(padded_fold))
        return padded_scaled_data

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

    def prepare_output_data(self, data):
        multi_out_data = []
        for fold in data:
            multi_out_data.append(self.make_multi_output_data(fold))
        return multi_out_data
