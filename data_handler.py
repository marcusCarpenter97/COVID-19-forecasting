import os
import matplotlib.pyplot as plt
import subprocess
import pandas as pd

# Copied from StackOverflow https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python/
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

# fetch the new data from the remote repo.
def get_github_updates():
    with cd("COVID-19"):
        subprocess.call(["git", "pull", "origin", "master"])

# read data from all three csv files.
def load_data():
   # confirmed = "time_series_19-covid-Confirmed.csv"
   # dead = "time_series_19-covid-Deaths.csv"
   # recovered = "time_series_19-covid-Recovered.csv"

    directory = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/"
    # The order of the file names is important in this list.
    files = ["time_series_covid19_confirmed_global.csv", "time_series_covid19_deaths_global.csv",
             "time_series_covid19_recovered_global.csv"]
    # Create a list of paths from the file names.
    paths = [os.path.join(directory, f) for f in files]

    data_frames = []
    # load all csv files into a list of data frames.
    for path in paths:
        df = pd.read_csv(path)
        data_frames.append(df)

    return data_frames # Confirmed, Dead and Recovered.

def load_updated_data():
    get_github_updates()
    return load_data()

# Infected people = confirmed - (dead + recovered) {the time series for each country/region must be summed for a total}
def calculate_total_infected(raw_data):
    confirmed, dead, recovered = raw_data
    c_sum = confirmed.sum()[2:] # The first two rows are the sums of Long and Lat which must be removed.
    d_sum = dead.sum()[2:]
    r_sum = recovered.sum()[2:]

    # Find the size of the smallest series.
    new_size = min(len(c_sum), len(d_sum), len(r_sum))

    # Trim the series so they all have the same size.
    c_sum = c_sum[:new_size].to_frame() 
    d_sum = d_sum[:new_size].to_frame() 
    r_sum = r_sum[:new_size].to_frame()  

    # r_sum uses m/dd/yyyy this needs to be converted to m/dd/yy to match the other two sets.
    r_sum = r_sum.set_index(c_sum.index)

    return (c_sum - (d_sum + r_sum)) # pandas DataFrame.

if __name__ == "__main__":
    raw_data = load_data()
    current_infected = calculate_total_infected(raw_data)
    print(current_infected)
    plot_df_line(current_infected)
    plt.show()

