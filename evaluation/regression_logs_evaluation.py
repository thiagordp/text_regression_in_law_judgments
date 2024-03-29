"""

@author Thiago Raulino Dal Pont
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLUMNS = [
    "index",
    "alg",
    "mse",
    "rmse",
    "r2",
    "adj_r2",
    "mae"
]

IGNORE_TECHS = [
    "linear_regression",
    "svr_poly_3",
    "svr_poly_2",
    "svr_poly_4",
    "svr_sigmoid",
    "svr_rbf",
    "mlp_100",
    "mlp_200",
    "sgd_regressor",
    "mlp_100_50_25",
    "random_forest_100",
    "random_forest_1000",
    "svr_linear",
    "mlp_200_100",
    "mlp_200_100_50",
    #"ridge",
    "random_forest_100_04_50",
    "random_forest_100_06_50",
    "random_forest_100_08_50",
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


def process_overfitting_log(log_path, description=""):
    df = pd.read_csv(log_path, index_col=0)

    # print(df.head())
    # print(df.describe())

    k_features = sorted([int(k) for k in df["k"].unique()])
    techs = df["tech"].unique()

    dict_rmse_test = dict()
    dict_r2_test = dict()
    dict_mae_test = dict()
    dict_rmse_ratio = dict()

    for tech in techs:
        rmse_test_results = list()
        r2_test_results = list()
        mae_test_results = list()
        rmse_ratio_results = list()

        for k in k_features:
            new_df = df[(df["k"] == k) & (df["tech"] == tech)]

            rmse_test_mean = np.mean(new_df["rmse_test"])
            r2_mean = np.mean(new_df["r2_test"])
            mae_mean = np.mean(new_df["mae_test"])
            rmse_ratio_mean = np.mean(new_df["rmse_ratio"])

            if k == 5000:
                print(tech, rmse_test_mean, r2_mean, sep="\t")
                #print("R2 Test\t", r2_mean)

            rmse_test_results.append([k, rmse_test_mean])
            r2_test_results.append([k, r2_mean])
            mae_test_results.append([k, mae_mean])
            rmse_ratio_results.append([k, rmse_ratio_mean])

        dict_r2_test[tech] = r2_test_results
        dict_rmse_test[tech] = rmse_test_results
        dict_mae_test[tech] = mae_test_results
        dict_rmse_ratio[tech] = rmse_ratio_results

    dest_file = log_path.replace(".csv", "_#.pdf")
    print(dest_file)
    plot_metrics(dict_r2_test, dest_file, "r2_test", "R² Test", description, y_range=[0, 0.8])
    plot_metrics(dict_rmse_test, dest_file, "rmse_test", "RMSE Test", description, y_range=[1500, 3200])
    plot_metrics(dict_rmse_ratio, dest_file, "rmse_ratio", "Ratio RMSE", description, y_range=[-0.2, 2])

    plot_metrics(dict_mae_test, dest_file, "mae_test", "MAE Test", description, y_range=[500, 3000])


def plot_metrics(dict_data, dest_file, metric, metrics_desc, description, y_range=[], x_range=[]):

    dest_file = dest_file.replace("#", metric)
    print(dest_file)
    plt.figure(figsize=(15, 10))
    plt.grid(linestyle=':')

    keys = sorted(dict_data.keys())

    markers = ["s", "X", "D", "o", "v", "^", "<", ">", "8"]
    i = -1
    for key in keys:
        i += 1
        if key in IGNORE_TECHS:
            continue

        data = dict_data[key]

        df = pd.DataFrame(data, columns=["k", metric])
        ks = list([str(x) for x in df["k"]])
        rmses = list(df[metric])

        marker = markers[i % len(markers)]
        plt.plot(ks, rmses, "-" + marker, label=key, linewidth=3, markersize=10)

    plt.xlabel('k', fontsize=18)
    if len(y_range) > 0:
        plt.ylim(top=y_range[1], bottom=y_range[0])

    plt.ylabel(metrics_desc.split()[0], fontsize=18)
    title_rmse = metrics_desc

    if description != "":
        title_rmse += " - " + description

    plt.title(title_rmse, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(bbox_to_anchor=(0.5, -0.1), ncol=4, loc='upper center', borderaxespad=0., fontsize=14)

    plt.tight_layout()
    plt.savefig(dest_file)
    plt.show()
