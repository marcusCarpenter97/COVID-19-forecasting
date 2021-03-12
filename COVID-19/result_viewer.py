import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

TARGET_DIR = "cross_val_results"

def build_file_name(model: str, reg: int, fold: int, e_type: str, o_type: str) -> str:
    return f"{model}_reg{reg}_fold{fold}_{e_type}_{o_type}"

def open_file(f_name: str):
    """
    Returns
    -------
    2D numpy array
    """
    path = os.path.join(TARGET_DIR, f_name)
    with np.load(path) as data:
        ret = [data[i] for i in data]  # There can be multiple arrays in a file.
    return np.squeeze(np.array(ret))

def rank_models(max_fold: int, output: list):
    """
    Parameters
    ----------
    max_fold: number of validation folds to be used.
    output: type of results to be used. Can be either 'val' or 'test'.
    """
    models = ["gru", "lstm"]
    reg_names = ["No reg", "L1", "L2", "Dropout", "L1L2", "All reg"]
    errors = ["rmse", "mae"]
    regs = range(len(reg_names))
    folds = range(max_fold+1)  # Range stops at max-1
    rank_bins = {}

    permutations = list(itertools.product(models, folds, errors, output))

    for perm in permutations:
        reg_compare = []
        for reg in regs:
            f_name = build_file_name(perm[0], reg, perm[1], perm[2], perm[3])
            reg_compare.append(open_file(f_name))  # TODO Exception if file name does not exist.
        reg_compare = np.stack(reg_compare)

        reg_mean = np.mean(np.mean(reg_compare, axis=2), axis=1)  # Take the mean twice because it's a 3D array.
        smallest_regs = np.where(reg_mean == reg_mean.min())[0]  # Get the indicies of all the occurances of the smallest value.

        try:
            rank_bins[f"{perm[0].upper()} {perm[2].upper()}"].append(smallest_regs)
        except KeyError:
            rank_bins[f"{perm[0].upper()} {perm[2].upper()}"] = [smallest_regs]

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    for axis, key in zip(axes.flatten(), rank_bins):
        rank_bins[key] = np.stack(rank_bins[key])
        reg_count = np.bincount(rank_bins[key].flatten())

        axis.bar(reg_names, reg_count)
        axis.set_yticks(range(max(reg_count)+1))  # Plus one because range stops at max minus one.
        axis.set_ylabel("Rank")
        axis.set_title(key)

        print(reg_count)

def root_mean_squared_scaled_error(orig, pred):
    # The average squared distance between two points in the data.
    denominator = (1/(len(orig)-1)) * np.sum(np.diff(orig, axis=1) ** 2, axis=1, keepdims=True)
    error = (orig - pred) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore divide by zero warnings
        scaled_error = (error / denominator)
        rmsse = np.sqrt(np.mean(scaled_error, axis=1))
    return rmsse

def merge_names_and_data(names, data):
    table = []
    for name, row in zip(names, data):
        table.append(list(row))
        table[-1].insert(0, name)
    return table

def make_rmsse_tables(lstm_reg: str, gru_reg: str, val_folds: int, test_data: str, loc_names: list) -> None:

    latex_tables = "latex_tables"
    orig = open_file(test_data)

    lstm_res = []
    gru_res = []

    for fold in inclusive_range(val_folds):
        lstm_file_name = f"lstm_reg{lstm_reg}_fold{fold}_test_pred"
        gru_file_name = f"gru_reg{gru_reg}_fold{fold}_test_pred"
        lstm_pred = open_file(lstm_file_name)
        gru_pred = open_file(gru_file_name)

        lstm_res.append(root_mean_squared_scaled_error(orig[fold], lstm_pred))
        gru_res.append(root_mean_squared_scaled_error(orig[fold], gru_pred))

        lstm_table = merge_names_and_data(loc_names, lstm_res[-1])
        gru_table = merge_names_and_data(loc_names, gru_res[-1])

        os.makedirs(latex_tables, exist_ok=True)
        lstm_path = os.path.join(latex_tables, f"{lstm_file_name}.tex")
        gru_path = os.path.join(latex_tables, f"{gru_file_name}.tex")

        with open(lstm_path, "w") as lt:
            lt.write(tabulate(lstm_table, tablefmt="latex_raw"))
        with open(gru_path, "w") as gt:
            gt.write(tabulate(gru_table, tablefmt="latex_raw"))

    avg_lstm = np.mean(lstm_res, axis=0)
    avg_gru = np.mean(gru_res, axis=0)

    assert(avg_lstm.shape == avg_gru.shape), f"{avg_lstm} and {avg_gru} should have the same shape. Got {avg_lstm.shape} and"
    "{avg_gru.shape}"

    avg_table = []
    for i in range(avg_lstm.shape[1]):
        avg_table.append(avg_lstm[:, i])
        avg_table.append(avg_gru[:, i])

    avg_table = np.stack(avg_table).T

    main_table = merge_names_and_data(loc_names, avg_table)
    print(main_table)
    main_path = os.path.join(latex_tables, f"avg_rmsse_lstm_reg{lstm_reg}_gru_reg_{gru_reg}.tex")
    temp = tabulate(main_table, tablefmt="latex_raw")
    print(temp)
    with open(main_path, "w") as mt:
        mt.write(temp)

def inclusive_range(stop, start=0, step=1):
    return range(start, stop+1, step)

def load_location_names():
    path = os.path.join(TARGET_DIR, "country_names.csv")
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        results = [row for row in reader]
    return results[0]

def handle_user_input():
    menu_text = """
Enter an option by its number:
1. Rank models using bar plots.
2. Create cross-validation tables.
3. Interactive prediction viewer.
4. Check validation folds for a location.
5. Exit.
"""
    min_val_fold = 0
    max_val_fold = 11
    min_reg = 0
    max_reg = 5

    val_res = "val"
    test_res = "test"

    params = {}

    while True:
        try:
            option = int(input(menu_text))
        except ValueError:
            print("Option must be an integer.")
            continue

        params["option"] = option

        if option == 1:
            val_fold = int(input(f"Validation fold index ({min_val_fold} to {max_val_fold}):"))  # TypeError
            out_type = input(f"Result type ({val_res} or {test_res}):")

            if val_fold in inclusive_range(max_val_fold, min_val_fold):
                params["fold"] = val_fold
            if out_type in (val_res, test_res):
                params["out"] = [out_type]

        elif option == 2:
            lstm_reg = int(input("Enter the number for the LSTM regularizer:\n0.No reg\n1.L1\n2.L2\n3.Dropout\n4.L1L2"
                                 " (ElasticNet)\n5.All reg\n"))
            gru_reg = int(input("Enter the number for the GRU regularizer:\n0.No reg\n1.L1\n2.L2\n3.Dropout\n4.L1L2"
                                " (ElasticNet)\n5.All reg\n"))
            val_fold = int(input(f"Validation fold index ({min_val_fold} to {max_val_fold}):"))  # TypeError

            if lstm_reg in inclusive_range(max_reg, min_reg):
                params["lstm_reg"] = lstm_reg
            if gru_reg in inclusive_range(max_reg, min_reg):
                params["gru_reg"] = gru_reg
            if val_fold in inclusive_range(max_val_fold, min_val_fold):
                params["val_fold"] = val_fold

        elif option == 5:
            raise SystemExit
        else:
            print("Invalid option.")
            continue

        return params

def main():

    test_data = "test_y"

    usr_input = handle_user_input()
    print(usr_input)  # For debugging.

    if usr_input["option"] == 1:
        rank_models(usr_input["fold"], usr_input["out"])

    if usr_input["option"] == 2:
        loc_names = load_location_names()
        make_rmsse_tables(usr_input["lstm_reg"], usr_input["gru_reg"], usr_input["val_fold"], test_data, loc_names)

if __name__ == "__main__":
    main()
    plt.show()
