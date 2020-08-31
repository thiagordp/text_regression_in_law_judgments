"""

@author
"""
import datetime

import numpy as np
import pandas as pd
from sklearn import metrics

from util.sheets_api import pull_sheet_data, SCOPES, SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME, export_data_to_sheets


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

            predictions_dict[int(sent_train)] = [sent_train, tech_train, "train", round(pred, 2), round((pred - actual), 2), round(actual, 2)]

        actual_values = row_test[1]
        pred_values = row_test[2]

        for k_tes in range(len(actual_values)):
            pred = pred_values[k_tes]
            actual = actual_values[k_tes]
            sent_test = sentences_test[k_tes]

            predictions_dict[int(sent_test)] = [int(sent_test), tech_train, "test", round(pred, 2), round((pred - actual), 2), round(actual, 2)]

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
        df = pd.DataFrame(data_tech.T, columns_tech).T

        list_columns.extend(columns_tech)
        list_dfs.append(df)

    result = pd.concat(list_dfs, axis=1)
    result.to_excel("results.xlsx", index=False)

    export_data_to_sheets(result, SCOPES, SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME)

