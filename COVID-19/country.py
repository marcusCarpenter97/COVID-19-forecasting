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

    def plot_country(self, bars=[]):
        ax = self.data[["Confirmed", "Deceased", "Recovered", "Infected"]].plot(title=f"{self.name}")
        ax.set_xlabel("Days")
        ax.set_ylabel("People")
        ax.legend(loc="upper left")

        if len(bars) > 0:
            for bar in bars:
                ax.axvline(bar, color="red", linestyle="--")

    def print_country(self):
        print(self.name)
        print(self.encoded_name)
        print(self.population)
        print(self.data)

    def diff_data(self):
        # The first row must be saved so the differencing can be undone.
        self.first_row = self.data.iloc[0]
        self.data = self.data.diff().dropna()

    def int_data(self):
        # TODO slow, make faster.
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

    def int_pred(self, predictions):
        """
        Rescales predictions by integrating values.

        Parameters
        predictions: numpy array.

        Returns
        numpy array.
        """
        # Get the original value for the row imediately before the prediction data.
        # This will be used to integrate the predictions.
        # Minus one becasuse of the 0 indexed arrays.
        idx = len(self.data) - len(predictions) - 1
        before_pred = self.data.iloc[idx].values

        def calc_row(before_pred, diffed):
            return before_pred + diffed.sum()
        res = [calc_row(before_pred, predictions[:row]) for row in range(1, len(predictions)+1)]
        return np.stack(res)

    def log_data(self):
        # Using the log(x+1) function to handle zeroes in the data.
        self.data = np.log1p(self.data)

    def exp_data(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                self.data = np.expm1(self.data)
            except RuntimeWarning as runw:
                print(f"RuntimeWarning: {runw}")
                print("The numbers given to exp where too big and caused an error. The data wasn't affected.")

    def split_data(self, test_size):
        """
        Split data into train, validation and test.

        Example:
        >>>data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        >>>data[:-s*2], data[-s*2:-s], data[-s:]
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 0])

        Parameters
        ----------
        test_size: int
            Size of validation and test set.
        """
        self.train_x, self.train_y, self.test = (self.data.values[:-test_size*2], self.data.values[-test_size*2:-test_size],
                self.data.values[-test_size:])

        self.train_x = self.train_x.reshape(1, self.train_x.shape[0], self.train_x.shape[1])
