import os
import csv
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

TARGET_DIR = "cross_val_results"
TEST_DATA = "test_y"
VAL_DATA = "val"

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="Display results  based on comand line arguments.")
    parser.add_argument("model", type=str, help="The model name (gru or lstm).")
    parser.add_argument("reg", type=int, help="The regulariser index to be used.")
    parser.add_argument("fold", type=int, help="The cross-validation fold to be used.")
    parser.add_argument("f_type", type=str, help="The file type to be used (rmse, hist, etc).")
    parser.add_argument("o_type", type=str, help="The type to be used (val or pred).")
    args = parser.parse_args()
    return args.model, args.reg, args.fold, args.f_type, args.o_type

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

def plot_pred(predictions, original, rmse_scores, mae_scores, loc_names):
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
        curr_loc = curr_loc % len(predictions)
        for idx, ax in enumerate(axes):
            ax.cla()
            ax.set_title(f"{sub_titles[idx]} / RMSE: {int(rmse_scores[curr_loc][idx])} / MAE: {int(mae_scores[curr_loc][idx])}")
            ax.plot(original[curr_loc].T[idx], color="b")
            ax.plot(predictions[curr_loc].T[idx], linestyle="--", color="orange")
        axes[0].set_ylabel("People")
        axes[1].set_xlabel("Days")
        axes[2].legend(["Original", "Predictions"], loc=0)
        fig.suptitle(loc_names[curr_loc])
        fig.canvas.draw()

    fig, axes = plt.subplots(ncols=predictions.shape[2], sharex=True, constrained_layout=True)
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

def handle_user_input(files, loc_names):
    try:
        option = int(input("Enter an option by its number:\n1. Compare models and make table.\n2. Plot validation vs test error"
                           "for a model.\n3. Plot error over validation folds for a model.\n4. Interactive prediction"
                           "viewer.\n"))
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
    elif option == 4:  # interactive prediction viewer.
        model = input("Model type (gru or lstm):")
        reg = input("Regularizer index (0 to 5):")
        fold = input("Validation fold index (0 to 8):")
        res_type = input("Result type (val or test):")
        pred_f_name = build_file_name(model, reg, fold, res_type, "pred")
        rmse_f_name = build_file_name(model, reg, fold, "rmse", res_type)
        mae_f_name = build_file_name(model, reg, fold, "mae", res_type)

        # Check if file names exist.
        assert(pred_f_name in files), f"{pred_f_name} does not exist."
        assert(rmse_f_name in files), f"{rmse_f_name} does not exist."
        assert(mae_f_name in files), f"{mae_f_name} does not exist."

        if res_type == "val":
            orig_data = open_file(VAL_DATA)
        elif res_type == "test":
            orig_data = open_file(TEST_DATA)
        else:
            print(f"Invalid result type must be val or test was {res_type}")

        try:
            original_fold = orig_data[int(fold)]  # Choose the correct validation fold.
        except ValueError:
            print("Fold index must be an integer.")
            raise SystemExit

        rmse_scores = open_file(rmse_f_name)
        mae_scores = open_file(mae_f_name)
        predictions = open_file(pred_f_name)

        plot_pred(predictions, original_fold, rmse_scores, mae_scores, loc_names)
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
