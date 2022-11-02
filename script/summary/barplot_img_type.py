import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

root_path = "../../output/logs/cnn_image/sampleNet/{}_{}_test.log"

m_color_types = {0: "blue"}


def format_value(val):
    val = float(str(val).strip())
    return "{:.2f}".format(round(val, 2))


def draw_plot(db_t, values_t):
    fig, ax = plt.subplots()

    plt.xlabel("Code Refactoring", fontsize=18, labelpad=12)
    plt.ylabel("Prediction Accuracy", fontsize=18, labelpad=10)

    y_ranges = list(np.arange(0, 100, 20)) + [100]
    plt.yticks(y_ranges, fontsize=14)
    plt.ylim([0, 100 + 0.02])

    x_ranges = ["Original", "Format", "Limit", "Window", "Redacted"]
    xx = ["{}".format(x_ranges[v]) for v in range(len(values_t[0]))]

    bar_objs = []
    for i, yy in enumerate(values_t):
        obj = ax.bar(xx, yy, width=0.5)
        bar_objs.append(obj)

    # ax.legend(loc=2)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.gca().tick_params(axis='both', which='major', pad=10)
    plt.gcf().subplots_adjust(left=0.125, bottom=0.15)
    plt.gcf().set_size_inches(8, 6)

    def auto_label(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * h,
                    '{} %'.format(format_value(h)),
                    fontsize=12, color="black",
                    ha='center', va='bottom')

    for bar_ in bar_objs:
        auto_label(bar_)

    save_file = "barplot_img_type_{}.png".format(db_t)
    if os.path.exists(save_file):
        os.remove(save_file)
    pathlib.Path(os.path.dirname(save_file)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300)
    plt.show()


def main():
    for db in ["top10"]:
        values_1 = []
        for ty in ["original", "reformat_base", "reformat_clipped", "reformat_window", "redacted_window"]:
            log_file = root_path.format(db, ty)
            last_val_acc = 0
            with open(log_file, 'r') as log_f:
                for line in log_f:
                    if "Overall Accuracy =" in str(line):
                        curr_val_acc = str(line).split("Overall Accuracy = ")[-1].strip()
                        curr_val_acc = float(curr_val_acc.split(" %")[0].strip())
                        last_val_acc = max(last_val_acc, curr_val_acc)
            values_1.append(last_val_acc)
        draw_plot(db, [values_1])
        print("Done: {}".format(db))
        print()


if __name__ == "__main__":
    main()
