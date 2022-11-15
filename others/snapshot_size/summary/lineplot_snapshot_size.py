import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

root_path = "../logs/sampleNet_{}_{}_{}.log"

m_img_types = {0: "Original", 1: "Refactor"}
m_color_types = {0: "red", 1: "green"}


def draw_plot(db_t, values_t):
    fig, ax = plt.subplots()

    plt.xlabel("Image Size", fontsize=18, labelpad=12)
    plt.ylabel("Prediction Accuracy", fontsize=18, labelpad=10)

    y_ranges = list(np.arange(0, 100, 20)) + [100]
    plt.yticks(y_ranges, fontsize=14)
    plt.ylim([0, 100 + 0.02])

    x_ranges = [32, 64, 128, 256, 512, 1024]
    xx = ["{}x{}".format(x_ranges[v], x_ranges[v]) for v in range(len(values_t[0]))]
    for i, yy in enumerate(values_t):
        ax.plot(xx, yy, 'ro-', color=m_color_types[i], label=m_img_types[i])

    ax.legend(loc=2, prop={'size': 14})
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.gca().tick_params(axis='both', which='major', pad=10)
    plt.gcf().subplots_adjust(left=0.125, bottom=0.15)
    plt.gcf().set_size_inches(8, 6)

    save_file = "lineplot_snapshot_size_{}.png".format(db_t)
    if os.path.exists(save_file):
        os.remove(save_file)
    pathlib.Path(os.path.dirname(save_file)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300)
    plt.show()


def main():
    debug_val = False
    for db in ["top10", "top50"]:
        values_1 = []
        for ty in ["original", "window"]:
            values_2 = []
            for sz in ["32", "64", "128", "256", "512", "1024"]:
                log_file = root_path.format(db, ty, sz)
                last_val_acc = 0
                with open(log_file, 'r') as log_f:
                    for line in log_f:
                        if debug_val:
                            if "val acc:" in str(line):
                                curr_val_acc = str(line).split(", val loss")[0].strip()
                                curr_val_acc = float(curr_val_acc.split("val acc: ")[-1].strip())
                                last_val_acc = max(last_val_acc, curr_val_acc)
                        else:
                            if "Overall Accuracy =" in str(line):
                                curr_val_acc = str(line).split("Overall Accuracy = ")[-1].strip()
                                curr_val_acc = float(curr_val_acc.split(" %")[0].strip())
                                last_val_acc = max(last_val_acc, curr_val_acc)
                values_2.append(last_val_acc)
            values_1.append(values_2)
        draw_plot(db, values_1)
        print("Done: {}".format(db))
        print()


if __name__ == "__main__":
    main()
