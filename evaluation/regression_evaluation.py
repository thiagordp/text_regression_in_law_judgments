"""

@author
"""
import datetime
import time

import numpy as np
from sklearn import metrics
import pandas as pd


def batch_evaluation(data, independent_vars=1):
    print()
    metrics_data = list()
    len_data = len(data)

    for row in data:
        key, y_test, y_pred = row

        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2_score = metrics.r2_score(y_test, y_pred)
        adjusted_r2_score = 1 - (((1 - r2_score) * (len_data - 1)) / (len_data - independent_vars - 1))
        mae = metrics.mean_absolute_error(y_test, y_pred)

        metrics_data.append([key, np.round(mse, 2), np.round(rmse, 2), np.round(r2_score, 4), np.round(adjusted_r2_score, 4), np.round(mae, 2)])

    df = pd.DataFrame(columns=["algorithm", "mse", "rmse", "r2", "adj_r2", "mae"], data=metrics_data)
    filename = "data/regression_metrics_" + str(datetime.datetime.today()).replace(":", "-").replace(".", "-") + ".csv"
    df.to_csv(filename.replace(" ", "_"))
