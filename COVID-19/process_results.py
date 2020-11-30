import os
import csv
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data

TARGET_DIR = "cross_val_results"

def filter_file_names(file_names, key_word):
    """
    Select file names that contain a key word in them.
    Parameters:
        file_names - list of flie names.
        key_word - string
    Retuns a list of the selected file names.
    """
    selected_names = []
    for file_name in file_names:
        if key_word in file_name:
            selected_names.append(file_name)
    return selected_names

def open_npz(file_names):
    """
    Opens .npz files.
    Parameters:
        file_names - list of file names.
    Returns a list containing the numpy arrays loaded from the files.
    """
    results = []
    for file_name in file_names:
        path = os.path.join(TARGET_DIR, file_name)
        temp = np.load(path)
        results.append(temp["arr_0"])
    return results

def compare_regularizers(file_names):
    """
    Parameters:
        file_names - list of file names.
    Returns a 3D numpy array containing the indicies for the regularizer with the lowest RMSE.
    """
    best_overall = []

    # For each fold
    for reg_name0, reg_name1, reg_name2, reg_name3, reg_name4, reg_name5 in zip(*file_names):
        reg_results = open_npz([reg_name0, reg_name1, reg_name2, reg_name3, reg_name4, reg_name5])
        # For each country
        best_fold = []
        for reg0, reg1, reg2, reg3, reg4, reg5 in zip(*reg_results):
            temp = np.stack([reg0, reg1, reg2, reg3, reg4, reg5])
            # Find the index of the smallest result as this maches the regularizer's index.
            best_fold.append(np.argmin(temp, axis=0))
        best_overall.append(np.stack(best_fold))

    return np.stack(best_overall)

def evaluate_regularizers(reg_comp):
    """
    reg_comp - 3D numpy array.
    """

    f_names = ["Confirmed:", "Deceased:", "Recovered:"]

    for fold_idx, fold in enumerate(reg_comp):
        print(f"Results for fold {fold_idx}:")
        for i, row in enumerate(fold.T):
            u, c = np.unique(row, return_counts=True)
            temp = dict(zip(u, c))
            # Get the highest result from the dictionary.
            best = max(temp.keys(), key=lambda k: temp[k])
            print(f"{f_names[i]} {temp} - Best regularizer: {best}")
        print()

def fill_missing_regs(data):
    reg_num = 6  # Number of regularizers used.
    formatted = []
    for fold_idx, fold in enumerate(data):
        temp = []
        for i, row in enumerate(fold.T):
            u, c = np.unique(row, return_counts=True)
            if len(u) < reg_num:
                missing_reg = sum(range(u[0],u[-1]+1)) - sum(u)
                u = np.insert(u, missing_reg, missing_reg)
                c = np.insert(c, missing_reg, 0)
            temp.append(c)
        formatted.append(temp)
    return np.stack(formatted)

def plot_gru_vs_lstm_tally_marks(gru_data, lstm_data):
    """
    Both params: numpy array of shape (folds, features, regs)
    """
    fig, axes = plt.subplots(gru_data.shape[0], gru_data.shape[1],  constrained_layout=True)
    for fold, gru_fold, lstm_fold in zip(axes, gru_data, lstm_data):
        for feature, gru_feat, lstm_feat in zip(fold, gru_fold, lstm_fold):
            feature.plot(gru_feat, marker="o")
            feature.plot(lstm_feat, marker="*")
    fig.legend(["GRU", "LSTM"], loc='upper center')
    plt.show()

def plot_gru_vs_lstm(gru_file, lstm_file):

    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    gru_error, lstm_error = open_npz([gru_file, lstm_file])

    countires = gru_error.shape[0]
    features = gru_error.shape[1]

    bar_idx = np.arange(countires)
    height = 0.5

    fig, axes = plt.subplots(1, features, constrained_layout=True)

    # Iterate over the countries
    for axis_idx, axis in enumerate(axes):
        axis.barh(bar_idx, gru_error.T[axis_idx], height=height, color="orange", log=True)
        axis.barh(bar_idx+height, lstm_error.T[axis_idx], height=height, color="blue", log=True)
        axis.set_title(sub_titles[axis_idx])

    fig.set_size_inches(8.27, 11.69)
    plt.savefig('foo.pdf', bbox_inches='tight')

def plot_rmse_over_folds(file_names, contry_idx):
    all_reg_errs = []
    for reg in file_names:
        all_reg_errs.append(np.stack([np.squeeze(open_npz([fold]))[contry_idx] for fold in reg]))
    all_reg_errs = np.stack(all_reg_errs)
    print(all_reg_errs.shape)

    fig, axes = plt.subplots(1, all_reg_errs.shape[-1], constrained_layout=True)
    for ax_idx, ax in enumerate(axes):
        for reg in all_reg_errs:
            print(reg)
            ax.plot(reg.T[ax_idx])

def plot_learning_curves(name):
    # Plot learning curves for the models.
    model_hist = filter_file_names(filter_file_names(files, name), "hist")
    print(model_hist)

    for hist in model_hist:
        path = os.path.join(TARGET_DIR, hist)
        df = pd.read_csv(path)
        df.plot(y=['output_1_root_mean_squared_error', 'output_2_root_mean_squared_error', 'output_3_root_mean_squared_error'],
                title=hist).get_figure().savefig(f"{hist}.png")

def plot_orig_vs_pred(country_idx):
    path = os.path.join(TARGET_DIR, "original_data")
    temp = np.load(path)
    for key in temp:
        print(key)
        print(temp[key].shape)
    fig, axis = plt.subplots(1, 3)
    for fold in temp["test_y"]:
        for feat, ax in zip(fold, axis):
            ax.plot(feat[country_idx])

def print_eval(name):
    gru_eval = filter_file_names(filter_file_names(files, "gru"), "eval")
    lstm_eval = filter_file_names(filter_file_names(files, "lstm"), "eval")

    o1 = "output_1_root_mean_squared_error"
    o2 = "output_2_root_mean_squared_error"
    o3 = "output_3_root_mean_squared_error"

    def truncate(number, digits=4) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    for i in range(7):
        for j in range(0, len(gru_eval), 7):
            gru_path = os.path.join(TARGET_DIR, gru_eval[i+j])
            lstm_path = os.path.join(TARGET_DIR, lstm_eval[i+j])
            with open(gru_path, "r") as g_file, open(lstm_path, "r") as l_file:
                g_data = json.load(g_file)
                l_data = json.load(l_file)

            print(f"{truncate(l_data[o1])} & {truncate(g_data[o1])} & {truncate(l_data[o2])} & {truncate(g_data[o2])} & {truncate(l_data[o3])} & {truncate(g_data[o3])}")

def save_country_names():
    d = data.Data()
    names = [country.name for country in d.countries]
    path = os.path.join(TARGET_DIR, "country_names.csv")
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(names)

def load_country_names():
    path = os.path.join(TARGET_DIR, "country_names.csv")
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        results = [row for row in reader]
    return results[0]

def setup(files, reg_names, model_name):
    files_names = filter_file_names(files, model_name)
    error_files = filter_file_names(files_names, "error")

    error_regs = []
    for reg_name in reg_names:
        error_regs.append(filter_file_names(error_files, reg_name))
    return error_regs

def find_best_regularizer(model):
    """
    Parameters:
        model - string - name of the model can be either 'lstm' or 'gru'.
    """
    pass

if __name__ == "__main__":

    save_country_names()
