import os
import platform
from datetime import datetime
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class cd:
    """
    Context manager for changing the current working directory.
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


DIRECTORY = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/"
# The order of the file names is important in this list.
FILES = ["time_series_covid19_confirmed_global.csv", "time_series_covid19_deaths_global.csv",
         "time_series_covid19_recovered_global.csv"]

def get_github_updates():
    """
    Fetch the new data from the remote repo.
    """
    with cd("COVID-19"):
        subprocess.call(["git", "pull", "origin", "master"])

def clone_git_repo():
    subprocess.call(["git", "clone", "https://github.com/CSSEGISandData/COVID-19.git"])

def load_data():
    """
    Read data from all csv files.
    """

    if not os.path.isdir(DIRECTORY):
        clone_git_repo()

    # Create a list of paths from the file names.
    paths = [os.path.join(DIRECTORY, f) for f in FILES]

    if platform.system() == 'Linux':  # TODO make cross-platform.
        stats = [os.stat(path) for path in paths]
        # Get today's date.
        today = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d')
        # Check whether all data files have been modified today. If not get updates.
        dates = [datetime.utcfromtimestamp(int(stat.st_mtime)).strftime('%Y-%m-%d') for stat in stats]
        if not all(today == creation_time for creation_time in dates):
            get_github_updates()

    data_frames = []
    # load all csv files into a list of data frames.
    for path in paths:
        df = pd.read_csv(path)
        data_frames.append(df)

    return data_frames # Confirmed, Dead and Recovered.

def load_updated_data():
    """
    Fetch GitHub updates before loading the data.
    """
    get_github_updates()
    return load_data()

def calculate_current_infected(data):
    """
    Infected people = confirmed - (dead + recovered) 
    The time series for each country/region must be summed for a total.
    """
    confirmed, dead, recovered = data
    c_sum = confirmed.sum(numeric_only=True)[2:]  # The first two rows are the sums of Long and Lat which must be removed.
    d_sum = dead.sum(numeric_only=True)[2:]
    r_sum = recovered.sum(numeric_only=True)[2:]

    # Find the size of the smallest series.
    new_size = min(len(c_sum), len(d_sum), len(r_sum))

    # Trim the series so they all have the same size.
    c_sum = c_sum[:new_size].to_frame() 
    d_sum = d_sum[:new_size].to_frame() 
    r_sum = r_sum[:new_size].to_frame()  

    # r_sum uses m/dd/yyyy this needs to be converted to m/dd/yy to match the other two sets.
    r_sum = r_sum.set_index(c_sum.index)

    return c_sum - (d_sum + r_sum) # pandas DataFrame.

def multivariate_to_supervised(data, steps):
    X, Y = [], []
    for idx in range(len(data)):
        offset = idx + steps
        try:
            x, y = data[idx:offset].values, data.iloc[offset].values
        except IndexError:
            break
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)

def series_to_supervised(data, before, after):
    new_cols = []
    
    for col in range(before, 0, -1):
        new_cols.append(data.shift(periods=col).rename(columns={0: -col}))

    new_cols.append(data)

    for col in range(1, after+1):
        new_cols.append(data.shift(periods=-col).rename(columns={0: col}))

    return pd.concat(new_cols, axis=1).dropna()

def prepare_data(data):
    # As the time series is non-stationary (the average changes over time) some data tranformations 
    # must take place. This will provide better results.

    # First the natural logarithm of the original data is taken.
    log_cases = pd.DataFrame(np.log(data))

    # Then the difference between each point in the data is calculated.
    first_diff = log_cases.diff().dropna()

    # As the first differencing did not provide good results the difference of the difference is taken.
    stationary_data = first_diff.diff().dropna()
    
    return stationary_data 

def invert_difference(orig_data, diff_data, interval=1):
    return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]

# Stationary to normal.
def rescale_data(data, first_diff, second_diff, log_sample, horizon_offset):
    undiff_two = invert_difference(second_diff[horizon_offset:], data)
    undiff_one = invert_difference(first_diff[horizon_offset+1:], undiff_two)
    log_data = invert_difference(log_sample[horizon_offset+2:], undiff_one)
    return np.exp(log_data)

def split_horizon(data, horizon):
    x = data.iloc[:, :horizon].to_numpy()
    y = data.iloc[:, horizon:].to_numpy()
    return x, y

def split_train_test(data, ratio):
    train_size = int(data.shape[0] * ratio)
    return data[:train_size], data[train_size:]

def adjust_data(data):
    log_cases = pd.DataFrame(np.log(data))
    first_diff = log_cases.diff().dropna()
    second_diff = first_diff.diff().dropna()
    stationary_data = second_diff.diff().dropna()
    return log_cases, first_diff, second_diff, stationary_data 

def plot(data_to_plot):
    fig = plt.figure() 
    plt.plot(data_to_plot)
    plt.xticks(rotation=90)
    return fig
