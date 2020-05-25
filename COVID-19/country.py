import numpy as np
import pandas as pd
import warnings


class Country:
    """ Represents a country with COVID-19 """

    def __init__(self, n, p, c, d, r, i, h):
        self.name = n
        self.population = p
        self.data = pd.concat([c, d, r, i, h], axis=1)
        self.data.columns=["Confirmed", "Deceased", "Recovered", "Infected", "Healthy"]

    def print_country(self):
        print(self.name)
        print(self.population)
        print(self.data)

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
