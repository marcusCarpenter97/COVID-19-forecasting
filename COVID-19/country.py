import pandas as pd

class Country:
    """ Represents a country with COVID-19 """

    def __init__(self, n, p, c, d, r):
        """
            Params:
                n - string - The country's name.
                p - int - The country's population.
                c - pandas Series - Time series of the confirmed cases.
                d - pandas Series - Time series of the deseased cases.
                r - pandas Series - Time series of the recovered cases.
        """
        self.name = n
        self.encoded_name = None
        self.population = p
        self.data = pd.concat([c, d, r], axis=1)
        self.data.columns=["Confirmed", "Deceased", "Recovered"]

    def find_divisor_offset(self, div):
        """
        Calculates an offset to make the data divisible by the horizon.
        By finding the greatest divisor of div that is smaller than num.

        div  - int - horizon
        returns offset - int

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
        # Find how many days have to be cut of the start to make the data divisible by the horizon.
        # This is required as the cross validation folds must the the same size.
        time_steps = len(self.data)
        return time_steps - ((time_steps//div)*div)

    def split_k_fold(self, train_size, horizon):
        """
        train_size - int - number of time steps in the train data.
        horizon - int - number of time steps to be forecasted.
        """

        # Make the data divisible by the horizon to guarantee all k folds have the same size.
        # Cuts out the old data instead of the new time steps.
        offset = self.find_divisor_offset(horizon)
        temp_data = self.data[offset:]

        val_end = train_size + horizon
        test_end = train_size + horizon * 2

        train = temp_data[:train_size]
        val = temp_data[train_size:val_end]
        test_x = temp_data[:val_end]
        test_y = temp_data[val_end:test_end]

        return train, val, test_x, test_y
