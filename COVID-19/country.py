import numpy as np
import pandas as pd
import warnings


class Country:
    """ Represents a country with COVID-19 """

    def __init__(self, n, p, c, d, r, i, h):
        self.name = n
        self.encoded_name = None
        self.population = p
        self.data = pd.concat([c, d, r, i, h], axis=1)
        self.data.columns=["Confirmed", "Deceased", "Recovered", "Infected", "Healthy"]

    def plot_country(self, train_date=None):
        ax = self.data[["Confirmed", "Deceased", "Recovered", "Infected"]].plot(title=f"{self.name}")
        ax.set_xlabel("Days")
        ax.set_ylabel("People")
        ax.legend(loc="upper left")
        if train_date:
            for i in train_date:
                ax.axvline(i, color="red", linestyle="--")

    def print_country(self):
        print(self.name)
        if self.encoded_name:
            print(self.encoded_name)
        print(self.population)
        print(self.data)

    def diff_data(self):
        # The first row must be saved so the differencing can be undone.
        self.first_row = self.data.iloc[0]
        self.data = self.data.diff().dropna()

    def int_data(self):
        def calc_row(diffed):
            return self.first_row + diffed.sum()
        res = [calc_row(self.data.iloc[:row]) for row in range(1, len(self.data)+1)]
        res.insert(0, self.first_row)
        self.data = pd.DataFrame(res)

        # The integration messes up the dates in the index column so they need to be fixed.
        self.data.reset_index(inplace=True, drop=True)
        self.data['days'] = pd.date_range(start=self.first_row.name, periods=len(self.data))
        self.data['days'] = self.data['days'].dt.strftime("%m/%d/%y")
        self.data.set_index('days', inplace=True)

    def log_data(self):
        self.data = np.log(self.data)

    def exp_data(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                self.data = np.exp(self.data)
            except RuntimeWarning as runw:
                print(f"RuntimeWarning: {runw}")
                print("The numbers given to exp where too big and caused an error. The data wasn't affected.")

    def find_divisor_offset(self, num, div):
        """
        Calculates an offset to make the data divisible by the horizon. 
        By finding the greatest divisor of div that is smaller than num.

        Example: If splitting the data into weeks.
        num = 137
        div = 7
        137/7 = 19.57... (Not divisible)
        Using the formula below:
        137 - (floor(137/7) * 7) = 4
        Four is the offset to make 137 divisible by 7 because
        137 - 4 = 133
        133/7 = 19 (Divisible)
        """
        return num - ((num//div)*div)

    def split_data(self, horizon, train_size):

        # Calculate the offset as the data will not always be divisible by the horizon.
        offset = self.find_divisor_offset(len(self.data), horizon)

        # Split the data.
        self.train, self.test = self.data.values[offset:offset+train_size], self.data.values[offset+train_size:]

        # Reshape it so that it is split into horizon sized chunks.
        self.train.reshape(self.train.shape[0]//horizon, horizon, self.train.shape[1])
        self.test.reshape(self.test.shape[0]//horizon, horizon, self.test.shape[1])

    def get_slice(self, start, end):
        return self.data.iloc[start:end]
