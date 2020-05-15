import os
import platform
from datetime import datetime
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COVID_DIR = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/"
# The order of the file names is important in this list.
FILES = ["time_series_covid19_confirmed_global.csv", "time_series_covid19_deaths_global.csv",
         "time_series_covid19_recovered_global.csv"]
TRAIN_SET_RATIO = 0.7

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

def pull_covid_updates():
    """
    Fetch the new data from the remote repo.
    Executed the git pull command.
    """
    with cd("COVID-19"):
        subprocess.call(["git", "pull", "origin", "master"])

def clone_covid_data():
    """
    Executes the git clone on the Johns Hopkins GitHub repository.
    """
    subprocess.call(["git", "clone", "https://github.com/CSSEGISandData/COVID-19.git"])

def check_for_updates(paths):
    """
    Updated the files if they have not been modified today.
    """
    if platform.system() == 'Linux':  # TODO make cross-platform.
        stats = [os.stat(path) for path in paths]
        # Get today's date.
        today = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d')
        # Get last modified date for each file.
        dates = [datetime.utcfromtimestamp(int(stat.st_mtime)).strftime('%Y-%m-%d') for stat in stats]
        # Check whether all data files have been modified today. If not get updates.
        if not all(today == creation_time for creation_time in dates):
            pull_covid_updates()

def load_population_data():
    """
    param: country_names - list of strings.
    """
    pop = pd.read_csv("kaggle_data/covid19-global-forecasting-week-5/train.csv")
    # Remove unnecessary columns.
    pop.drop(columns=["Id", "County", "Province_State", "Weight", "Date", "Target", "TargetValue"], inplace=True)
    # Find country's population.
    return pop.groupby(['Country_Region']).max()

def load_covid_data():
    """
    Read data from all csv files.
    """

    # First time loding the data.
    if not os.path.isdir(COVID_DIR):
        clone_covid_data()

    # Create a list of paths from the file names.
    paths = [os.path.join(COVID_DIR, f) for f in FILES]

    # Get new data.
    check_for_updates(paths)

    # Load all csv files into a list of data frames.
    data_frames = [pd.read_csv(path) for path in paths]

    # Remove unnecessary columns and return dataframes.
    return [df.drop(columns=['Province/State', 'Lat', 'Long']) for df in data_frames]  # Confirmed, Dead and Recovered.

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

def split_train_test(data):
    train_size = int(data.shape[0] * TRAIN_SET_RATIO)
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
