import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

TARGET_DIR = "cross_val_results"
TEST_DATA = "test_y"
VAL_DATA = "val"

def build_file_name(model, reg, fold, f_type, o_type):
    """
    Parameters
    ----------
    reg : int
    model : str
    fold : int
    f_type : str
    o_type : str
    """
    return f"{model}_reg{reg}_fold{fold}_{f_type}_{o_type}"

def open_file(f_name):
    """
    Parameters
    ----------
    f_name : str

    Returns
    -------
    ret : array of numpy arrays
    """
    path = os.path.join(TARGET_DIR, f_name)
    with np.load(path) as data:
        ret = [data[i] for i in data]  # There can be multiple arrays in a file.
    return np.squeeze(np.array(ret))

def plot_error(data):
    fig, axis = plt.subplots(ncols=data.shape[1], constrained_layout=True)
    print(data)
    for idx, ax in enumerate(axis):
        ax.plot(data.T[idx])
        ax.set_yscale("log")
    plt.draw()

def load_location_names():
    path = os.path.join(TARGET_DIR, "country_names.csv")
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        results = [row for row in reader]
    return results[0]

def plot_pred(original, loc_names, gru, lstm):
    curr_loc = 0
    def key_event_handler(event):
        sub_titles = ["Confirmed", "Deceased", "Recovered"]
        nonlocal curr_loc
        if event.key == "right":
            curr_loc += 1
        elif event.key == "left":
            curr_loc -= 1
        else:
            return
        curr_loc = curr_loc % len(gru["preds"])  # Assumes gru and lstm have same size.
        for idx, ax in enumerate(axes):
            ax.cla()
            ax.set_title(f"{sub_titles[idx]}")
            ax.plot(original[curr_loc].T[idx], color="b")
            ax.plot(gru["preds"][curr_loc].T[idx], linestyle="--", color="orange")
            ax.plot(lstm["preds"][curr_loc].T[idx], linestyle="-.", color="r")
        axes[0].set_ylabel("People")
        axes[1].set_xlabel("Days")
        axes[2].legend(["Original", "GRU predictions", "LSTM predictions"], loc=0)
        fig.suptitle(loc_names[curr_loc])
        fig.canvas.draw()

        print(f"{loc_names[curr_loc]}")
        print(f"GRU  | C: RMSE {int(gru['rmse'][curr_loc][0])} / MAE {int(gru['mae'][curr_loc][0])} | D: RMSE "
              f"{int(gru['rmse'][curr_loc][1])} / MAE {int(gru['mae'][curr_loc][1])} | R: RMSE {int(gru['rmse'][curr_loc][2])}"
              f" / MAE {int(gru['mae'][curr_loc][2])}")
        print(f"LSTM | C: RMSE {int(lstm['rmse'][curr_loc][0])} / MAE {int(lstm['mae'][curr_loc][0])} | D: RMSE "
              f"{int(lstm['rmse'][curr_loc][1])} / MAE {int(lstm['mae'][curr_loc][1])} | R: RMSE {int(lstm['rmse'][curr_loc][2])}"
              f" / MAE {int(lstm['mae'][curr_loc][2])}")

    fig, axes = plt.subplots(ncols=gru["preds"].shape[2], sharex=True, constrained_layout=True)
    fig.suptitle("Press the left or right arrow key to begin.")
    fig.canvas.mpl_connect("key_press_event", key_event_handler)

def compare_models():
    """
    latex : bool
    """
    models = ["lstm", "gru"]
    regs = 6
    reg_names = ["No reg", "L1", "L2", "Dropout", "L1L2", "All reg"]
    folds = np.arange(9)
    errors = ["rmse", "mae"]
    outputs = ["val", "test"]
    amper = " & "  # For a latex table
    hist_data = {"lstm_rmse": [], "gru_rmse": [], "lstm_mae": [], "gru_mae": []}

    permutations = itertools.product(models, folds, errors, outputs)
    for perm in permutations:
        reg_comp = []
        for reg in np.arange(regs):
            f_name = build_file_name(perm[0], reg, perm[1], perm[2], perm[3])
            reg_comp.append(open_file(f_name))  # TODO Exception if file name does not exist.
        reg_comp = np.stack(reg_comp)
        reg_comp = reg_comp.T.argmin(axis=2)  # transpose it to compare the regularizers.
        reg_comp = np.apply_along_axis(np.bincount, 1, reg_comp, minlength=regs).argmax(axis=1)  # count the best regularizers.

        hist_data[f"{perm[0]}_{perm[2]}"].append(reg_comp)
        print(f"Best for {perm}")
        # Translate the reg indicies to their names then convert the list to a string.
        print(f"{amper.join(map(str, [reg_names[i] for i in reg_comp]))}")
    for key in hist_data:
        hist_data[key] = np.stack(hist_data[key])
        temp_count = np.bincount(hist_data[key].flatten())
        fig, ax = plt.subplots(constrained_layout=True)
        ax.bar(reg_names, temp_count)
        ax.set_yticks(np.arange(temp_count[np.argmax(temp_count)]+1))
        ax.set_title(key)
    print(hist_data)

def plot_val_v_test(model, reg, fold):
    errors = ["rmse", "mae"]
    outs = ["val", "test"]
    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    permutations = itertools.product(errors, outs)
    f_names = [build_file_name(model, reg, fold, perm[0], perm[1]) for perm in permutations]
    for i in range(0, len(f_names), len(errors)):
        val_data = open_file(f_names[i])
        test_data = open_file(f_names[i+1])
        fig, axes = plt.subplots(ncols=val_data.shape[1], constrained_layout=True)
        for idx, axis in enumerate(axes):
            axis.plot(val_data.T[idx], color="b")
            axis.plot(test_data.T[idx], linestyle="--", color="orange")
            axis.set_title(sub_titles[idx])
            axis.set_yscale("log")
        axes[0].set_ylabel("Error")
        axes[1].set_xlabel("Locations")
        axes[2].legend(["Validation", "Test"], loc=0)
        fig.suptitle(f"{model} / regularizer: {reg} / validation fold: {fold}")

def plot_val_folds(reg):
    models = ["lstm", "gru"]
    errors = ["rmse", "mae"]
    outs = ["val", "test"]
    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    hist_data = {"lstm_rmse_val": [], "lstm_rmse_test": [], "gru_rmse_val": [], "gru_rmse_test": [], "lstm_mae_val": [],
                 "lstm_mae_test": [], "gru_mae_val": [], "gru_mae_test": []}
    folds = np.arange(9)
    permutations = itertools.product(models, folds, errors, outs)
    for perm in permutations:
        f_name = build_file_name(perm[0], reg, perm[1], perm[2], perm[3])
        hist_data[f"{perm[0]}_{perm[2]}_{perm[3]}"].append(np.mean(open_file(f_name)))
    fig, axis = plt.subplots(constrained_layout=True)
    axis.plot(hist_data["lstm_rmse_val"])
    axis.plot(hist_data["lstm_rmse_test"])
    axis.plot(hist_data["gru_rmse_val"])
    axis.plot(hist_data["gru_rmse_test"])
    axis.legend(["LSTM val", "LSTM test", "GRU val", "GRU test"])
    axis.set_title("RMSE")
    plt.draw()
    fig, axis = plt.subplots(constrained_layout=True)
    axis.plot(hist_data["lstm_mae_val"])
    axis.plot(hist_data["lstm_mae_test"])
    axis.plot(hist_data["gru_mae_val"])
    axis.plot(hist_data["gru_mae_test"])
    axis.legend(["LSTM val", "LSTM test", "GRU val", "GRU test"])
    axis.set_title("MAE")
    plt.draw()
    for key in hist_data:
        print(f"{key}: {hist_data[key]}")

def root_mean_squared_scaled_error(orig, pred):
    # The average squared distance between two points in the data.
    denominator = (1/(len(orig)-1)) * np.sum(np.diff(orig, axis=1) ** 2, axis=1, keepdims=True)
    error = (orig - pred) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore divide by zero warnings
        scaled_error = (error / denominator)
        rmsse = np.sqrt(np.mean(scaled_error, axis=1))
    return rmsse

def calculate_rmsse_corss_val(model, reg):
    folds = np.arange(9)
    orig = open_file(TEST_DATA)
    res = []
    for fold in folds:
        file_name = f"{model}_reg{reg}_fold{fold}_test_pred"
        pred = open_file(file_name)
        res.append(root_mean_squared_scaled_error(orig[fold], pred))
    return res

def make_cross_val_plots(res, loc_names):
    # TODO plot of errors over validations folds.
    pass

def merge_names_and_data(names, data):
    table = []
    for name, row in zip(names, data):
        table.append(list(row))
        table[-1].insert(0, name)
    return table

def make_cross_val_table(res, loc_names):
    for i, fold in enumerate(res):
        print(f"RMSSE for validation fold: {i}")
        table = merge_names_and_data(loc_names, fold)
        print(tabulate(table, tablefmt="latex_raw"))

def handle_user_input(files, loc_names):
    try:
        option = int(input("Enter an option by its number:\n1. Compare models and make table.\n2. Plot validation vs test error"
                           "for a model.\n3. Plot error over validation folds for a model.\n4. Make cross-validation table.\n5."
                           " Interactive prediction viewer.\n"))
    except ValueError:
        print("Option must be an integer.")
        raise SystemExit

    if option == 1:
        compare_models()
    elif option == 2:
        model = input("Model type (gru or lstm):")
        reg = input("Regularizer index (0 to 5):")
        fold = input("Validation fold index (0 to 8):")
        plot_val_v_test(model, reg, fold)
    elif option == 3:
        reg = input("Regularizer index (0 to 5):")
        plot_val_folds(reg)
    elif option == 4:
        model = input("Model type (gru or lstm):")
        reg = input("Regularizer index (0 to 5):")
        rmsse_res = calculate_rmsse_corss_val(model, reg)
        make_cross_val_table(rmsse_res, loc_names)
        make_cross_val_plots(rmsse_res, loc_names)
    elif option == 5:  # interactive prediction viewer.
        gru_reg = input("GRU regularizer index (0 to 5):")
        lstm_reg = input("LSTM regularizer index (0 to 5):")
        fold = input("Validation fold index (0 to 8):")
        res_type = input("Result type (val or test):")

        gru_pred_f_name = build_file_name("gru", gru_reg, fold, res_type, "pred")
        gru_rmse_f_name = build_file_name("gru", gru_reg, fold, "rmse", res_type)
        gru_mae_f_name = build_file_name("gru", gru_reg, fold, "mae", res_type)

        lstm_pred_f_name = build_file_name("lstm", lstm_reg, fold, res_type, "pred")
        lstm_rmse_f_name = build_file_name("lstm", lstm_reg, fold, "rmse", res_type)
        lstm_mae_f_name = build_file_name("lstm", lstm_reg, fold, "mae", res_type)

        # Check if file names exist.
        assert(gru_pred_f_name in files), f"{gru_pred_f_name} does not exist."
        assert(gru_rmse_f_name in files), f"{gru_rmse_f_name} does not exist."
        assert(gru_mae_f_name in files), f"{gru_mae_f_name} does not exist."

        assert(lstm_pred_f_name in files), f"{lstm_pred_f_name} does not exist."
        assert(lstm_rmse_f_name in files), f"{lstm_rmse_f_name} does not exist."
        assert(lstm_mae_f_name in files), f"{lstm_mae_f_name} does not exist."

        if res_type == "val":
            orig_data = open_file(VAL_DATA)
        elif res_type == "test":
            orig_data = open_file(TEST_DATA)
        else:
            print(f"Invalid result type must be val or test was {res_type}")
            raise SystemExit

        try:
            original_fold = orig_data[int(fold)]  # Choose the correct validation fold.
        except ValueError:
            print("Fold index must be an integer.")
            raise SystemExit

        gru = {}
        gru["rmse"] = open_file(gru_rmse_f_name)
        gru["mae"] = open_file(gru_mae_f_name)
        gru["preds"] = open_file(gru_pred_f_name)

        lstm = {}
        lstm["rmse"] = open_file(lstm_rmse_f_name)
        lstm["mae"] = open_file(lstm_mae_f_name)
        lstm["preds"] = open_file(lstm_pred_f_name)

        plot_pred(original_fold, loc_names, gru, lstm)
    else:
        print("Invalid option.")
        raise SystemExit

def main():
    files = os.listdir(TARGET_DIR)
    loc_names = load_location_names()
    handle_user_input(files, loc_names)

if __name__ == "__main__":
    main()
    plt.show()
