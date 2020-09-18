import numpy as np
import pandas as pd
import warnings


class Country:
    """ Represents a country with COVID-19 """

    def __init__(self, n, p, c, d, r, i, h):
        """
            Params:
                n - string - The country's name.
                p - int - The country's population.
                c - pandas Series - Time series of the confirmed cases.
                d - pandas Series - Time series of the deseased cases.
                r - pandas Series - Time series of the recovered cases.
                i - pandas Series - Time series of the infected cases.
                h - pandas Series - Time series of the healthy cases.
        """
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
        print(f"Name: {self.name}")
        print(f"Encoded name: {self.encoded_name}")
        print(f"Population: {self.population}")
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

    # TODO ?
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

    def split_data(self, test_size):
        """
            Split the data into training and testing sets for the model.
            Params:
                test_size - int - size of the test data to be used a an index for the split.
        """
        self.train, self.test = self.data.values[:-test_size], self.data.values[-test_size:]

    # TODO unsused.
    def supervise_data2(self, horizon):
        """
        Convert the training data into a supervised set.
        Many to one.
        """
        x, y = [], []
        for i in range(len(self.train)):
            end = i + horizon
            if end < len(self.train):
                x.append(self.train[i:end, :])
                y.append(self.train[end, :])
        # Stack all sub arrays.
        self.train_x = np.stack(x)
        self.train_y = np.stack(y)
        self.train_y = self.train_y.reshape(self.train_y.shape[0], 1, self.train_y.shape[1])

    # TODO !
    def supervise_data(self, horizon):
        """
        Convert the training data into a supervised set.
        Many to many.
        """
        start = 0
        x, y = [], []
        # For each data point.
        for _ in range(len(self.train)):
            # Calculate offsets for end of in and out samples.
            end = start + horizon
            out_end = end + horizon
            # Check if output offset is still within the data.
            if out_end <= len(self.train):
                # Copy in and out sections fron data.
                x.append(self.train[start:end])
                y.append(self.train[end:out_end])
            # Mode to the naxt step.
            start += 1
        # Stack all sub arrays into one 2d array.
        self.train_x = np.stack(x)
        self.train_y = np.stack(y)

    # TODO unsused.
    def get_slice(self, start, end):
        return self.data.iloc[start:end]
