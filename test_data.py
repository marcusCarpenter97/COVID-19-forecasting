import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_handler

# difference dataset
def difference(data, interval):
        return [data[i] - data[i - interval] for i in range(interval, len(data))]

# invert difference
def invert_difference(orig_data, diff_data, interval=1):
        return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]

def reverse_difference(original, diff_predictions):
    diff_predictions.reset_index(drop=True, inplace=True)  # Convert the inde column from dates to ints.
    return pd.DataFrame([original + sum(diff_predictions[:t+1][0]) for t, _ in diff_predictions.iterrows()]) # un-diff predictions

def plot(data_to_plot):
    print(f"{data_to_plot}")
    print(type(data_to_plot))

    fig = plt.figure() 
    plt.plot(data_to_plot)
    return fig

if __name__ == "__main__":
    raw_data = data_handler.load_data()

    current_infected = data_handler.calculate_total_infected(raw_data)
    print(f"og   {len(list(current_infected[0]))}")
    original_fig = plot(current_infected)

    log_cases = pd.DataFrame(np.log(current_infected))
    print(f"log   {len(list(log_cases[0]))}")
    log_fig = plot(log_cases)

    first_diff = log_cases.diff().dropna()
    print(f"diff1   {len(list(first_diff[0]))}")
    first_fig = plot(first_diff)

    stationary_data = first_diff.diff().dropna()
    print(f"stat   {len(list(stationary_data[0]))}")
    stationary_fig = plot(stationary_data)

    station_to_diff = pd.DataFrame(invert_difference(list(first_diff[0]), list(stationary_data[0])))
    print(f"to_diff   {len(list(station_to_diff[0]))}")
    to_diff_fig = plot(station_to_diff)

    diff_to_log = pd.DataFrame(invert_difference(list(log_cases[0][1:]), list(station_to_diff[0])))
    print(f"to_log   {len(list(diff_to_log[0]))}")
    to_log_fig = plot(diff_to_log)

    log_to_og = pd.DataFrame(np.exp(diff_to_log))
    print(f"to_og   {len(list(log_to_og[0]))}")
    to_og_fig = plot(log_to_og)

    print("Original vs log_to_og")
    print(pd.concat([current_infected.reset_index(drop=1).add_suffix('_1'),
                    log_to_og.reset_index(drop=1).add_suffix('_2')], axis=1).fillna(''))

    print("Log vs diff_to_log")
    print(pd.concat([log_cases.reset_index(drop=1).add_suffix('_1'),
                    diff_to_log.reset_index(drop=1).add_suffix('_2')], axis=1).fillna(''))

    print("First diff vs station_to_diff")
    print(pd.concat([first_diff.reset_index(drop=1).add_suffix('_1'),
                    station_to_diff.reset_index(drop=1).add_suffix('_2')], axis=1).fillna(''))

    plt.show()

