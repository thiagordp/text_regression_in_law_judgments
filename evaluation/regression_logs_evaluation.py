"""

@author Thiago Raulino Dal Pont
"""
import glob
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLUMNS = [
    "index",
    "alg",
    "mse",
    "rmse",
    "r2",
    "adj_r2",
    "mae"
]


def get_mean_metrics(values_dict):
    final_values = dict()

    for key in values_dict.keys():
        values = values_dict[key]

        df = pd.DataFrame(data=values, columns=COLUMNS)

        alg = key

        values = np.array(df["mse"].values)
        mse = np.mean(values)

        values = np.array(df["rmse"].values)
        rmse = np.mean(values)

        values = np.array(df["r2"].values)
        r2 = np.mean(values)

        values = np.array(df["mae"].values)
        mae = np.mean(values)

        final_values[alg] = [mse, rmse, r2, mae]

    return final_values


def plot_metrics(type, mean_dict):
    x = list(mean_dict.keys())

    mse = list()
    rmse = list()
    r2 = list()
    mae = list()

    for alg in mean_dict.keys():
        mse.append(mean_dict[alg][0])
        rmse.append(mean_dict[alg][1])
        r2.append(mean_dict[alg][2])
        mae.append(mean_dict[alg][3])

    plt.plot(x, rmse)
    plt.show()


def process_log(log_path):
    print("-------------------------------------------------")
    print(log_path)
    list_log_files = glob.glob(log_path + "*.csv")

    # os.remove()

    list_dfs = list()
    for log_file in list_log_files:
        log_df = pd.read_csv(log_file)
        list_dfs.append(log_df)

    concat_df = pd.concat(list_dfs)
    concat_df.to_csv(log_path + "merge_log.csv")

    alg_list = concat_df["algorithm"].unique()

    values_dict = dict()

    for alg in alg_list:
        values = concat_df.loc[concat_df['algorithm'] == alg].values
        values_dict[alg] = values

    mean_dict = get_mean_metrics(values_dict)

    plot_metrics(log_path, mean_dict)
