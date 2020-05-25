import pandas as pd

class Country:
    """ Represents a country with COVID-19 """

    def __init__(self, n, p, c, d, r, i, h):
        #print(c, d, r, i, h)
        self.name = n
        self.population = p
        self.data = pd.concat([c, d, r, i, h], axis=1)
        self.data.columns=["Confirmed", "Deceased", "Recovered", "Infected", "Healthy"]

    def print_country(self):
        print(self.name)
        print(self.population)
        print(self.data)
