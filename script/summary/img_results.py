from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def format_value(val):
    return "{:.2f}".format(round(val * 100, 2))


def overall_results():
    for model_ in img_models:
        for db_ in img_datasets:
            for type_ in img_types:
                test_file = img_root_path.format(type_, db_, model_)
                if not Path(test_file).is_file():
                    continue

                df_test = pd.read_csv(test_file)

                labels = np.array(df_test["label"].tolist())
                predictions = np.array(df_test["prediction"].tolist())
                results_dct = classification_report(y_true=labels, y_pred=predictions, output_dict=True)

                P = format_value(results_dct[avg_metric]['precision'])
                R = format_value(results_dct[avg_metric]['recall'])
                F1 = format_value(results_dct[avg_metric]['f1-score'])
                A = format_value(results_dct['accuracy'])

                print("{}-{}-{}: P={}, R={}, F1={}, A={}".format(model_, db_, type_, P, R, F1, A))


if __name__ == "__main__":
    img_root_path = "../../output/predict/cnn_image/{}/{}_{}_test.csv"
    img_types = ["original", "reformat_window", "redacted_window"]
    img_datasets = ["top10", "top50"]
    img_models = ["alexNet", "resNet"]
    avg_metric = "weighted avg"  # "macro avg"
    overall_results()
