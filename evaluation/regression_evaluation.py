"""

@author
"""
import datetime

import numpy as np
import pandas as pd
from sklearn import metrics


def process_row_metric(row):
    key, y_test, y_pred = row

    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return [key, np.round(mse, 2), np.round(rmse, 2), np.round(r2_score, 4), np.round(mae, 2)]


def batch_evaluation(results_train, results_test, sentence_train, sentences_test, description=""):
    metrics_test = list()
    metrics_train = list()

    for row in results_test:
        metrics_test.append(process_row_metric(row))

    columns = ["algorithm", "mse", "rmse", "r2", "mae"]
    df = pd.DataFrame(columns=columns, data=metrics_test)
    filename = "data/regression_metrics_test_" + str(datetime.datetime.today()).replace(":", "-").replace(".", "-") + "_" + description + ".csv"
    df.to_csv(filename.replace(" ", "_"))

    list_predictions = list()
    for i in range(len(results_train)):
        row_train = results_train[i]
        row_test = results_test[i]

        tech_train = row_train[0]
        tech_test = row_test[0]

        actual_values = row_train[1]
        pred_values = row_train[2]

        for j_tr in range(len(pred_values)):
            pred = pred_values[j_tr]
            actual = actual_values[j_tr]
            sent_train = sentence_train[j_tr]

            list_predictions.append(["train", tech_train, sent_train, pred, actual, abs(pred - actual)])

        actual_values = row_test[1]
        pred_values = row_test[2]

        for k_tes in range(len(actual_values)):
            pred = pred_values[k_tes]
            actual = actual_values[k_tes]
            sent_test = sentences_test[k_tes]

            list_predictions.append(["test", tech_train, sent_test, pred, actual, abs(pred - actual)])

        df = pd.DataFrame(list_predictions, columns=["type", "technique", "sentence", "prediction", "actual", "abs_error"])

        df.to_csv("data/predictions/" + tech_test + ".csv")
        df.to_excel("data/predictions/" + tech_test + ".xlsx")
