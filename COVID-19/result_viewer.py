import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def build_file_name(fold, reg, ens, model, error, partition):
    return f"fold_{fold}_reg_{reg}_ens_{ens}_{model}_{error}_{partition}"

def load_npz(file_name):
    result_dir = "cross_val_results"

    file_path = os.path.join(result_dir, file_name)
    with np.load(file_path) as data:
        ret = [data[i] for i in data]  # There can be multiple arrays in a file.
    return np.squeeze(np.array(ret))

def format_errors(fold, model, error, partition):
    ensemble_size = 10
    regularizers = 6
    results = []
    for regularizer in range(regularizers):
        mean_error = []
        for trial in range(ensemble_size):
            file_name = build_file_name(fold, regularizer, trial, model, error, partition)
            model_errors = load_npz(file_name)
            mean_error.append(np.mean(model_errors))
        results.append(mean_error)
    return np.stack(results).T

def make_box_plot(lstm, gru, fold):
    def draw_plot(data, offset,edge_color, fill_color):
        pos = np.arange(data.shape[1])+offset
        bp = ax.boxplot(data, positions= pos, widths=0.3, showmeans=True, meanline=True, patch_artist=True, manage_ticks=False)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        return bp

    fig, ax = plt.subplots()
    bpA = draw_plot(lstm, -0.2, "tomato", "white")
    bpB = draw_plot(gru, +0.2,"skyblue", "white")
    plt.xticks(range(6))

    ax.set_xticklabels(["No reg", "L1", "L2", "Dropout", "ElasticNet", "All regs"])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Comparison of models for validation fold {fold}")
    ax.legend([bpA["boxes"][0], bpB["boxes"][0]], ["LSTM", "GRU"])

    fig_path = os.path.join("plots", f"boxplot_{fold}")
    plt.savefig(fig_path)

def make_comparison_table(gru, lstm, fold):
    header = ["Model", "Type", "No reg", "L1", "L2", "Dropout", "ElasticNet", "All regs"]
    with open("model_comparison.csv", "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        gru_mean_row = ["GRU", "Mean", *list(np.mean(gru, axis=0))]
        gru_var_row = ["GRU", "Var", *list(np.var(gru, axis=0))]
        lstm_mean_row = ["LSTM", "Mean", *list(np.mean(lstm, axis=0))]
        lstm_var_row = ["LSTM", "Var", *list(np.var(lstm, axis=0))]
        writer.writerow(gru_mean_row)
        writer.writerow(gru_var_row)
        writer.writerow(lstm_mean_row)
        writer.writerow(lstm_var_row)

def make_box_plots():
    folds = 14
    for fold in range(folds):
        gru_errors = format_errors(fold, "gru", "rmse", "test")
        lstm_errors = format_errors(fold, "lstm", "rmse", "test")
        make_box_plot(lstm_errors, gru_errors, fold)
        make_comparison_table(lstm_errors, gru_errors, fold)

def load_location_names():
    path = os.path.join("cross_val_results", "country_names.csv")
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        results = [row for row in reader]
    return results[0]

def make_ensemble_plots(fold, reg, model, partition):
    preds = []
    errs = []
    ensemble_size = 10
    loc_names = load_location_names()
    display = (0, 10, 11)
    sub_titles = ["Confirmed", "Deceased", "Recovered"]
    orig = load_npz("test_y")

    for e in range(ensemble_size):
        pred_file_name = build_file_name(fold, reg, e, model, partition, "pred")
        error_file_name = build_file_name(fold, reg, e, model, "rmse", partition)
        preds.append(load_npz(pred_file_name))
        errs.append(load_npz(error_file_name))
    avg_pred = np.mean(preds, axis=0)
    avg_errs = np.mean(errs, axis=0)

    fig, axes = plt.subplots(1,3)
    for loc_idx, loc_name in enumerate(loc_names):
        for idx, axis in enumerate(axes):
            for p in preds:
                axis.plot(p[loc_idx].T[idx], color="moccasin", label="Ensemble")
            axis.plot(orig[fold][loc_idx].T[idx], color="blue", label="Original")
            axis.plot(avg_pred[loc_idx].T[idx], linestyle="--", color="red", label=f"Average {model.upper()}")
            axis.set_title(f"{sub_titles[idx]} : {round(avg_errs[loc_idx][idx], 3)} RMSE")
        handles, labels = axes[2].get_legend_handles_labels()
        axes[0].set_ylabel("People")
        axes[1].set_xlabel("Days")
        axes[2].legend([handle for i, handle in enumerate(handles) if i in display],
                       [label for i, label in enumerate(labels) if i in display], loc=0)
        fig.suptitle(loc_name)
        fig_path = os.path.join("plots", f"fold_{fold}_reg_{reg}_{model}_{partition}_{loc_name}")
        plt.savefig(fig_path)
        plt.cla()

if __name__ == "__main__":
    make_box_plots()
    #fold = 13
    #reg = 0
    #model = "gru"
    #partition = "test"
    #make_ensemble_plots(fold, reg, model, partition)
