
class Country:
    """ Represents a country with COVID-19 """

    def __init__(self, n, p, c=None, d=None, r=None):
        self.name = n
        self.popultion = p['Population']
        self.confirmed = c.sum(numeric_only=True)
        self.deceased = d.sum(numeric_only=True)
        self.recovered = r.sum(numeric_only=True)
        self.infected = None

    def print_country(self):
        print(self.name)
        print(self.popultion)
        print(self.confirmed)
        print(self.deceased)
        print(self.recovered)
        print(self.infected)
