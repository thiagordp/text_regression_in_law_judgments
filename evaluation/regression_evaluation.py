"""

@author
"""
import datetime

import numpy as np
import pandas as pd
from sklearn import metrics

from util.sheets_api import pull_sheet_data, SCOPES, SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME


def process_row_metric(row):
    key, y_test, y_pred = row

    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_score = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)

    return [key, mse, rmse, r2_score, mae]


def get_cross_validation_average(result_list):
    new_list = list()

    new_results = list()
    for fold in result_list:
        for row in fold:
            new_results.append(row)

    df = pd.DataFrame(new_results, columns=["tech", "rmse_test", "rmse_train", "rmse_ratio",
                                            "r2_train", "r2_test", "r2_ratio",
                                            "mae_train", "mae_test", "mae_ratio", "k"])
    list_tech = list(df["tech"].unique())

    for tech in list_tech:
        rmse_test = np.mean(df[df["tech"] == tech]["rmse_test"])
        rmse_train = np.mean(df[df["tech"] == tech]["rmse_train"])
        rmse_ratio = np.mean(df[df["tech"] == tech]["rmse_ratio"])
        r2_test = np.mean(df[df["tech"] == tech]["r2_test"])
        r2_train = np.mean(df[df["tech"] == tech]["r2_train"])
        r2_ratio = np.mean(df[df["tech"] == tech]["r2_ratio"])
        mae_test = np.mean(df[df["tech"] == tech]["mae_test"])
        mae_train = np.mean(df[df["tech"] == tech]["mae_train"])
        mae_ratio = np.mean(df[df["tech"] == tech]["mae_ratio"])
        k = round(np.mean(df[df["tech"] == tech]["k"]))

        new_list.append([tech, rmse_train, rmse_test, rmse_ratio, r2_train, r2_test, r2_ratio, mae_train, mae_test, mae_ratio, k])

    return new_list


def percentage_error(y_pred, y_test):
    if abs(y_pred) < 0.1 and y_test == 0:
        return 0

    if y_test == 0:
        y_test = 0.1

    return (y_pred - y_test) / y_test


def compare_results(metrics_train, metrics_test):
    techs = list(metrics_train["algorithm"].unique())
    techs.extend(list(metrics_test["algorithm"]))

    techs = set(techs)
    results = list()
    for tech in techs:
        m_train = metrics_train[metrics_train["algorithm"] == tech]
        m_test = metrics_test[metrics_test["algorithm"] == tech]

        rmse_train = float(np.mean(m_train["rmse"]))
        rmse_test = float(np.mean(m_test["rmse"]))
        try:
            rmse_ratio = (rmse_test / rmse_train) - 1
        except:
            rmse_ratio = 2 ** 32

        r2_train = float(np.mean(m_train["r2"]))
        r2_test = float(np.mean(m_test["r2"]))
        try:
            r2_ratio = (r2_test / r2_train) - 1
        except:
            r2_ratio = 2 ** 32
        mae_train = float(np.mean(m_train["mae"]))
        mae_test = float(np.mean(m_test["mae"]))
        try:
            mae_ratio = (mae_test / mae_train) - 1
        except:
            mae_ratio = 2 ** 32

        results.append([tech, rmse_train, rmse_test, rmse_ratio, r2_train, r2_test, r2_ratio, mae_train, mae_test, mae_ratio])

    return results


def overfitting_evaluation(results_train, results_test):
    metrics_test = list()
    metrics_train = list()

    for row in results_test:
        metrics_test.append(process_row_metric(row))
    for row in results_train:
        metrics_train.append(process_row_metric(row))
    columns = ["algorithm", "mse", "rmse", "r2", "mae"]

    df_test = pd.DataFrame(columns=columns, data=metrics_test)
    df_train = pd.DataFrame(columns=columns, data=metrics_train)

    return compare_results(df_train, df_test)


def overfitting_prediction(sentence_test_list, test_predictions_list):
    dict_predictions = dict()

    dict_predictions["sentence"] = list(sentence_test_list)

    for test_predictions in test_predictions_list:
        tech = test_predictions[0]
        actual = test_predictions[1]
        pred = test_predictions[2]

        if tech in dict_predictions.keys():
            dict_predictions[tech].extend(pred)
        else:
            dict_predictions[tech] = list(pred)

    df = pd.DataFrame(dict_predictions)
    print(df.describe())
    print(df.columns)
    print(df.head(10))

    file_name = "data/overfitting/predictions_binary_k_150.csv"
    df.to_csv(file_name, index=False)
    df.to_excel(file_name.replace(".csv", ".xlsx"), index=False)


def batch_evaluation(results_train, results_test, sentence_train, sentences_test, description=""):
    metrics_test = list()
    metrics_train = list()

    for row in results_test:
        metrics_test.append(process_row_metric(row))
    for row in results_train:
        metrics_train.append(process_row_metric(row))
    columns = ["algorithm", "mse", "rmse", "r2", "mae"]

    df_test = pd.DataFrame(columns=columns, data=metrics_test)
    df_train = pd.DataFrame(columns=columns, data=metrics_train)

    # compare_results(df_train, df_test)

    filename = "data/regression_metrics_test_" + str(datetime.datetime.today()).replace(":", "-").replace(".", "-") + "_" + description + ".csv"
    df_test.to_csv(filename.replace(" ", "_"))

    list_predictions = dict()

    set_techs = set()
    for i in range(len(results_train)):
        predictions_dict = dict()
        row_train = results_train[i]
        row_test = results_test[i]

        tech_train = row_train[0]
        tech_test = row_test[0]

        set_techs.add(tech_train)

        actual_values = row_train[1]
        pred_values = row_train[2]

        for j_tr in range(len(pred_values)):
            pred = pred_values[j_tr]
            actual = actual_values[j_tr]
            sent_train = sentence_train[j_tr]

            predictions_dict[int(sent_train)] = [sent_train, tech_train, "train", round(pred, 2), round(percentage_error(pred, actual), 4), round(actual, 2)]

        actual_values = row_test[1]
        pred_values = row_test[2]

        for k_tes in range(len(actual_values)):
            pred = pred_values[k_tes]
            actual = actual_values[k_tes]
            sent_test = sentences_test[k_tes]

            predictions_dict[int(sent_test)] = [int(sent_test), tech_train, "test", round(pred, 2), round(percentage_error(pred, actual), 4), round(actual, 2)]

        list_predictions[tech_train] = predictions_dict

        # df = pd.DataFrame(list_predictions, columns=["sentence", "technique", "type", "prediction", "abs_error", "actual"])
        # df.sort_values(by=['sentence'], inplace=True)
        # df.to_csv("data/predictions/" + description + "_" + tech_test + ".csv")
        # df.to_excel("data/predictions/" + description + "_" + tech_test + ".xlsx")

    data = pull_sheet_data(SCOPES, SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME)
    df_sheets = pd.DataFrame(data[1:], columns=data[0])

    sentences_ids_sheets = df_sheets["Sentença"].to_list()

    list_types = []
    formatted_predictions = dict()

    for tech in list_predictions.keys():
        tech_predictions = list_predictions[tech]
        for sent_id in sentences_ids_sheets:
            int_sent = int(sent_id)

            try:
                sent, _, type, pre, err, act = tech_predictions[int_sent]
                list_types.append(type)
                try:
                    formatted_predictions[tech].count(1)
                except:
                    formatted_predictions[tech] = list()

                formatted_predictions[tech].append([pre, err])

            except:
                formatted_predictions[tech].append([0, 0])
                list_types.append("")

    list_dfs = list()
    list_columns = list()

    df_types = pd.DataFrame(data=list_types, columns=["type"])
    list_dfs.append(df_sheets)
    list_dfs.append(df_types)

    for tech in formatted_predictions:
        data_tech = formatted_predictions[tech]
        columns_tech = [tech, tech + " Error"]
        data_tech = np.array(data_tech)
        df_test = pd.DataFrame(data_tech.T, columns_tech).T

        list_columns.extend(columns_tech)
        list_dfs.append(df_test)

    result = pd.concat(list_dfs, axis=1)
    result.to_excel("results.xlsx", index=False)

    # export_data_to_sheets(result, SCOPES, SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME)


def save_predictions(pred_test, y_test, sentence_test, pred_traint, y_train, sentence_train):

    return None
