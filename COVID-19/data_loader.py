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
