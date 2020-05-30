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
            ax.axvline(train_date, color="red", linestyle="--")

    def print_country(self):
        print(self.name)
        if self.encoded_name:
            print(self.encoded_name)
        print(self.population)
        print(self.data)

    def diff_data(self):
        # The first row must be saved so the differencing can be undone.
        self.first_row = self.data.iloc[0]
        self.data = self.data.diff()

    def int_data(self):
        def calc_row(diffed):
            return self.first_row + diffed.sum()
        res = [calc_row(self.data.iloc[:row]) for row in range(1, len(self.data)+1)]
        self.data = pd.DataFrame(res)

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

    def get_slice(self, start, end):
        return self.data.iloc[start:end]
