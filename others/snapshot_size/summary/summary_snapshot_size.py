import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

root_path = "../predict/{}/{}_sampleNet_test_{}.csv"

avg_metrics = {
    "weighted": "weighted avg",
    "macro": "macro avg"
}
avg_metric = "weighted"


def format_value(val):
    return "{:.2f}".format(round(val * 100, 2))


def main():
    for ty in ["original", "window"]:
        for db in ["top10", "top50"]:
            for sz in ["32", "64", "128", "256", "512", "1024"]:
                test_file = root_path.format(ty, db, sz)
                df_test = pd.read_csv(test_file)
                labels = np.array(df_test["label"].tolist())
                predictions = np.array(df_test["prediction"].tolist())
                results_dct = classification_report(y_true=labels, y_pred=predictions, output_dict=True)

                P = format_value(results_dct[avg_metrics[avg_metric]]['precision'])
                R = format_value(results_dct[avg_metrics[avg_metric]]['recall'])
                F1 = format_value(results_dct[avg_metrics[avg_metric]]['f1-score'])
                A = format_value(results_dct['accuracy'])

                print("{}-{}-{}: ".format(ty, db, sz), P, R, F1, A)
        print()


if __name__ == "__main__":
    main()
