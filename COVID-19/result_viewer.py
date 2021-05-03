import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_errors(fold, reg, ens, model, error, partition):
    result_dir = "cross_val_results"
    file_name = f"fold_{fold}_reg_{reg}_ens_{ens}_{model}_{error}_{partition}"

    file_path = os.path.join(result_dir, file_name)
    with np.load(file_path) as data:
        return data["arr_0"]  # This assumes all files are contained in arr_0. Possible KeyError.

def format_errors(model, error, partition):
    ensemble_size = 10
    regularizers = 6
    results = []
    for regularizer in range(regularizers):
        mean_error = []
        for trial in range(ensemble_size):
            model_errors = load_errors(0, regularizer, trial, model, error, partition)
            mean_error.append(np.mean(model_errors))
        results.append(mean_error)
    return np.stack(results).T

def make_box_plot(dataA, dataB, fold):
    def draw_plot(data, offset,edge_color, fill_color):
        pos = np.arange(data.shape[1])+offset
        bp = ax.boxplot(data, positions= pos, widths=0.3, showmeans=True, meanline=True, patch_artist=True, manage_ticks=False)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        return bp

    fig, ax = plt.subplots()
    bpA = draw_plot(dataA, -0.2, "tomato", "white")
    bpB = draw_plot(dataB, +0.2,"skyblue", "white")
    plt.xticks(range(6))

    ax.set_xticklabels(["No reg", "L1", "L2", "Dropout", "ElasticNet", "All regs"])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Comparison of models for validation fold {fold}")
    ax.legend([bpA["boxes"][0], bpB["boxes"][0]], ["LSTM", "GRU"])

if __name__ == "__main__":
    gru_errors = format_errors("gru", "rmse", "test")
    lstm_errors = format_errors("lstm", "rmse", "test")
    make_box_plot(gru_errors, lstm_errors, 0)

    plt.show()
