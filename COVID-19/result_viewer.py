import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def build_file_name(model: str, reg: int, fold: int, e_type: str, o_type: str) -> str:
    return f"{model}_reg{reg}_fold{fold}_{e_type}_{o_type}"

def open_file(f_name: str):
    """
    Returns
    -------
    2D numpy array
    """
    path = os.path.join("cross_val_results", f_name)
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

def make_rmse_table(lstm_reg: str, gru_reg: str, val_folds: int, loc_names: list) -> None:

    latex_tables = "latex_tables"

    gru_rmse = open_file(f"gru_reg{gru_reg}_fold{val_folds}_rmse_test")
    gru_mae = open_file(f"gru_reg{gru_reg}_fold{val_folds}_mae_test")
    lstm_rmse = open_file(f"lstm_reg{lstm_reg}_fold{val_folds}_rmse_test")
    lstm_mae = open_file(f"lstm_reg{lstm_reg}_fold{val_folds}_mae_test")

    assert(gru_rmse.shape == gru_mae.shape), f"gru_rmse and gru_mae should have the same shape. Got {gru_rmse.shape} and "
    "{gru_mae.shape}"
    assert(lstm_rmse.shape == lstm_mae.shape), f"lstm_rmse and lstm_mae should have the same shape. Got {lstm_rmse.shape} and "
    "{lstm_mae.shape}"

    gru_merged = []
    for col in range(gru_rmse.shape[1]):
        gru_merged.append(gru_rmse[:, col])
        gru_merged.append(gru_mae[:, col])

    gru_merged = np.stack(gru_merged).T

    lstm_merged = []
    for col in range(lstm_rmse.shape[1]):
        lstm_merged.append(lstm_rmse[:, col])
        lstm_merged.append(lstm_mae[:, col])

    lstm_merged = np.stack(lstm_merged).T

    table_merged = []
    for row in range(gru_merged.shape[0]):
        table_merged.append(gru_merged[row])
        table_merged.append(lstm_merged[row])

    table_merged = np.stack(table_merged)

    models = []
    names = []
    for row in range(table_merged.shape[0]):
        if row % 2 == 0:
            models.append("GRU")
            names.append(loc_names.pop(0))
        else:
            models.append("LSTM")
            names.append("")

    temp_table = merge_names_and_data(models, table_merged)
    final_table = merge_names_and_data(names, temp_table)

    main_path = os.path.join(latex_tables, f"rmse_lstm_reg{lstm_reg}_gru_reg_{gru_reg}_fold{val_folds}.tex")

    with open(main_path, "w") as mt:
        mt.write(tabulate(final_table, tablefmt="latex_raw"))

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

    main_path = os.path.join(latex_tables, f"avg_rmsse_lstm_reg{lstm_reg}_gru_reg_{gru_reg}.tex")

    with open(main_path, "w") as mt:
        mt.write(tabulate(main_table, tablefmt="latex_raw"))

def interactive_viewer(original, gru_preds, lstm_preds, val_fold, loc_names):
    curr_loc = -1

    def key_event_handler(event):
        sub_titles = ["Confirmed", "Deceased", "Recovered"]
        nonlocal curr_loc
        if event.key == "right":
            curr_loc += 1
        elif event.key == "left":
            curr_loc -= 1
        else:
            return
        curr_loc = curr_loc % len(gru_preds)  # Assumes gru and lstm have same size.

        for idx, axis in enumerate(axes):
            axis.cla()
            axis.set_title(f"{sub_titles[idx]}")
            axis.plot(original[curr_loc].T[idx], color="b")
            axis.plot(gru_preds[curr_loc].T[idx], linestyle="--", color="orange")
            axis.plot(lstm_preds[curr_loc].T[idx], linestyle="-.", color="r")
        axes[0].set_ylabel("People")
        axes[1].set_xlabel("Days")
        axes[2].legend(["Original", "GRU predictions", "LSTM predictions"], loc=0)
        fig.suptitle(f"{loc_names[curr_loc]} - validation fold: {val_fold}")
        fig.canvas.draw()

    fig, axes = plt.subplots(ncols=gru_preds.shape[2], sharex=True, constrained_layout=True)
    fig.suptitle("Press the left or right arrow key to begin.")
    fig.canvas.mpl_connect("key_press_event", key_event_handler)

def view_validation_folds(original, gru_preds, lstm_preds, val_fold, loc_index, loc_name):
    curr_val_fold = -1

    def key_event_handler(event):
        sub_titles = ["Confirmed", "Deceased", "Recovered"]
        nonlocal curr_val_fold
        if event.key == "right":
            curr_val_fold += 1
        elif event.key == "left":
            curr_val_fold -= 1
        else:
            return
        curr_val_fold = curr_val_fold % (val_fold+1)  # Inclusive

        for idx, axis in enumerate(axes):
            axis.cla()
            axis.set_title(f"{sub_titles[idx]}")
            axis.plot(original[curr_val_fold][loc_index].T[idx], color="b")
            axis.plot(gru_preds[curr_val_fold][loc_index].T[idx], linestyle="--", color="orange")
            axis.plot(lstm_preds[curr_val_fold][loc_index].T[idx], linestyle="-.", color="r")
        axes[0].set_ylabel("People")
        axes[1].set_xlabel("Days")
        axes[2].legend(["Original", "GRU predictions", "LSTM predictions"], loc=0)
        fig.suptitle(f"{loc_name} - validation fold: {curr_val_fold}")
        fig.canvas.draw()

    fig, axes = plt.subplots(ncols=3, sharex=True, constrained_layout=True)
    fig.suptitle("Press the left or right arrow key to begin.")
    fig.canvas.mpl_connect("key_press_event", key_event_handler)

def inclusive_range(stop, start=0, step=1):
    return range(start, stop+1, step)

def load_location_names():
    path = os.path.join("cross_val_results", "country_names.csv")
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
4. Plot validation folds for a location.
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

        if option not in (1,2,3,4,5):
            print("Invalid option.")
            continue

        if option == 1:
            val_fold = int(input(f"Validation fold index ({min_val_fold} to {max_val_fold}):"))  # TypeError
            out_type = input(f"Result type ({val_res} or {test_res}):")

            if val_fold in inclusive_range(max_val_fold, min_val_fold):
                params["fold"] = val_fold
            if out_type in (val_res, test_res):
                params["out"] = [out_type]

        if option in (2, 3, 4):
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

        if option == 4:
            params["loc_name"] = input("Enter the name of the desired location:")

        if option == 5:
            raise SystemExit

        return params

def main():

    test_data = "test_y"

    usr_input = handle_user_input()
    print(usr_input)  # For debugging.
    loc_names = load_location_names()

    if usr_input["option"] == 1:
        rank_models(usr_input["fold"], usr_input["out"])

    if usr_input["option"] == 2:
        make_rmsse_tables(usr_input["lstm_reg"], usr_input["gru_reg"], usr_input["val_fold"], test_data, loc_names)
        make_rmse_table(usr_input["lstm_reg"], usr_input["gru_reg"], usr_input["val_fold"], loc_names)

    if usr_input["option"] == 3:
        orig = open_file(test_data)
        gru_preds = open_file(f"gru_reg{usr_input['gru_reg']}_fold{usr_input['val_fold']}_test_pred")
        lstm_preds = open_file(f"lstm_reg{usr_input['lstm_reg']}_fold{usr_input['val_fold']}_test_pred")
        interactive_viewer(orig[usr_input["val_fold"]], gru_preds, lstm_preds, usr_input["val_fold"], loc_names)

    if usr_input["option"] == 4:
        try:
            loc_index = loc_names.index(usr_input["loc_name"])
        except ValueError:
            print(f"{usr_input['loc_name']} is not a valid location. Try again.")
            raise SystemExit
        orig = open_file(test_data)[:usr_input['val_fold']+1]  # Plus one to make the slice inclusive.
        gru_preds = []
        lstm_preds = []
        for fold in inclusive_range(usr_input['val_fold']):
            gru_preds.append(open_file(f"gru_reg{usr_input['gru_reg']}_fold{fold}_test_pred"))
            lstm_preds.append(open_file(f"lstm_reg{usr_input['lstm_reg']}_fold{fold}_test_pred"))
        view_validation_folds(orig, gru_preds, lstm_preds, usr_input["val_fold"], loc_index, usr_input["loc_name"])

if __name__ == "__main__":
    main()
    plt.show()
