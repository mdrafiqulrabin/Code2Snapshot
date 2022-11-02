import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def format_value(val):
    return "{:.2f}".format(round(val * 100, 2))


def overall_results():
    for model_ in tok_models:
        for db_ in tok_datasets:
            for type_ in tok_types:
                test_file = tok_root_path.format(model_, db_, type_)
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
    tok_root_path = "../../output/predict/{}_token/{}_{}_test.csv"
    tok_types = ["kind", "value", "xvalue", "literal", "xliteral"]
    tok_datasets = ["top10", "top50"]
    tok_models = ["bilstm"]
    avg_metric = "weighted avg"  # "macro avg"
    overall_results()
