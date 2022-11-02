import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


root_path = "../../output/predict/c2x_path/prediction_{}_java_{}_test_best.txt"


def format_value(val):
    return "{:.2f}".format(round(val * 100, 2))


def overall_results():
    for db_ in ["top10", "top50"]:
        for model_ in ["code2vec", "code2seq"]:
            test_file = root_path.format(model_, db_)
            df_test = pd.read_csv(test_file)

            labels = np.array(df_test["method"].tolist())
            predictions = np.array(df_test["predict"].tolist())
            results_dct = classification_report(y_true=labels, y_pred=predictions, output_dict=True)

            P = format_value(results_dct[avg_metric]['precision'])
            R = format_value(results_dct[avg_metric]['recall'])
            F1 = format_value(results_dct[avg_metric]['f1-score'])
            A = format_value(results_dct['accuracy'])

            print("{}-{}: P={}, R={}, F1={}, A={}".format(model_, db_, P, R, F1, A))


if __name__ == "__main__":
    avg_metric = "weighted avg"  # "macro avg"
    overall_results()
