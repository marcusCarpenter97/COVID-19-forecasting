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

    def split_data(self, test_size, horizon):

        # Calculate the offset as the data will not always be divisible by the horizon.
        offset = self.find_divisor_offset(len(self.data), horizon)

        # Split the data.
        self.train, self.test = self.data.values[offset:-test_size], self.data.values[-test_size:]

        # Reshape it so that it is split into horizon sized chunks.
        #self.train.reshape(self.train.shape[0]//horizon, horizon, self.train.shape[1])
        self.test = self.test.reshape(self.test.shape[0]//horizon, horizon, self.test.shape[1])

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
