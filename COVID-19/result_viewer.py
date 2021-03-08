import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

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
    folds = range(max_fold)
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

def handle_user_input():
    menu_text = """
Enter an option by its number:
1. Rank models using bar plots.
2. Create cross-validation tables.
3. Interactive prediction viewer.
4. Check validation folds for a location.
5. Exit.
"""
    while True:
        try:
            option = int(input(menu_text))
        except ValueError:
            print("Option must be an integer.")
            continue

        if option == 5:
            raise SystemExit

        if option in range(1,5):
            return option
        else:
            print("Invalid option.")

def main():
    usr_chiose = handle_user_input()
    print(usr_chiose)

if __name__ == "__main__":
    #main()
    rank_models(12, ["test"])
    plt.show()
