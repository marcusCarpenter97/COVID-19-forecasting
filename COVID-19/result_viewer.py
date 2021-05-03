import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

data1 = np.random.rand(10)
print(f"Mean: {np.mean(data1)} Variance: {np.var(data1)}")
data2 = np.random.rand(10)
print(f"Mean: {np.mean(data2)} Variance: {np.var(data2)}")
data3 = np.random.rand(10)
print(f"Mean: {np.mean(data3)} Variance: {np.var(data3)}")
data4 = np.random.rand(10)
print(f"Mean: {np.mean(data4)} Variance: {np.var(data4)}")
data5 = np.random.rand(10)
print(f"Mean: {np.mean(data5)} Variance: {np.var(data5)}")
data6 = np.random.rand(10)
print(f"Mean: {np.mean(data6)} Variance: {np.var(data6)}")
data7 = np.random.rand(10)
print(f"Mean: {np.mean(data7)} Variance: {np.var(data7)}")
data8 = np.random.rand(10)
print(f"Mean: {np.mean(data8)} Variance: {np.var(data8)}")
data9 = np.random.rand(10)
print(f"Mean: {np.mean(data9)} Variance: {np.var(data9)}")
data10 = np.random.rand(10)
print(f"Mean: {np.mean(data10)} Variance: {np.var(data10)}")
data11 = np.random.rand(10)
print(f"Mean: {np.mean(data11)} Variance: {np.var(data11)}")
data12 = np.random.rand(10)
print(f"Mean: {np.mean(data12)} Variance: {np.var(data12)}")

dataA = np.stack([data1, data3, data5, data7, data9, data11]).T
dataB = np.stack([data2, data4, data6, data8, data10, data12]).T

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
ax.set_title("Comparison of models for validation fold X")
ax.legend([bpA["boxes"][0], bpB["boxes"][0]], ["LSTM", "GRU"])

plt.show()
