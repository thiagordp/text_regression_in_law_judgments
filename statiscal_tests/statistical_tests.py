"""
Statistical Tests for Regression results

"""
import time

import numpy as np
import pandas as pd
from scipy import stats

DICT_ATTR_NAME = {
    "fs": "Feature\nSelection",
    "or1": "Outliers\nRemoval\n(Train)",
    "or2": "Outliers\nRemoval\n(All)",
    "cv": "Cross-\nValidation",
    "at": "Addition\nof\nAELE",
    "ng": "N-Grams\nExtraction",
    "oa": "Overfitting\nAvoidance"
}

DICT_TECH_NAME = {
    "adaboost": "Adaboost",
    "bagging": "Bagging",
    "decision_tree": "Decision Tree",
    "elastic_net": "Elastic Net",
    "ensemble_voting_bg_mlp_gd_xgb": "Ensemble Voting",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "Neural Network",
    "random_forest": "Random Forest",
    "ridge": "Ridge",
    "svr_poly_rbf": "SVM (RBF)",
    "xgboost": "XGBoosting",
    "elapsed_time": "Elapsed Time"
}

DICT_FORMAT_METRIC = {
    "R2": "{:,.2f}",
    "RMSE": "{:,.0f}"
}


def get_data_for_test(df_data=pd.DataFrame(), attr_col_name=str(), result_col_name=str()):
    aux = df_data[df_data[attr_col_name] == 0]
    x2 = aux[result_col_name]

    aux = df_data[df_data[attr_col_name] == 1]
    x1 = aux[result_col_name]

    return x1, x2


def main():
    # Load dataset
    metrics = ["MAE", "R2", "RMSE", "elapsed"]
    alternative = {
        "MAE": "less",  # MAE w/o Adj is greater than MAE w/ adjustments.
        "R2": "greater",
        "RMSE": "less",
        "elapsed": "less"
    }

    data_results = []

    df_binary_table = pd.read_excel("data/paper/final_analysis/binary_table.xlsx", sheet_name=0)
    independent_cols = df_binary_table.columns[:7]
    dependent_cols_all = sorted(df_binary_table.columns[7:])
    count = 0

    t1 = time.time()

    for metric in metrics:
        dependent_cols = [col for col in dependent_cols_all if str(col).startswith(metric)]

        for indep_col in independent_cols:
            for dep_col in dependent_cols:
                x1, x2 = get_data_for_test(df_binary_table, indep_col, dep_col)

                n1 = len(x1)
                n2 = len(x2)
                mean1 = np.median(x1)
                mean2 = np.median(x2)
                stat1, p = stats.mannwhitneyu(x1, x2, alternative=alternative[metric])
                stat2 = n1 * n2 - stat1
                z_score = (stat1 - ((n1 * n2) / 2)) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)

                dep_col = dep_col.split()[-1].strip()
                data_results.append(
                    [
                        metric,
                        DICT_ATTR_NAME[indep_col],
                        DICT_TECH_NAME[dep_col],
                        n1, n2,
                        mean1, mean2,
                        stat1,
                        round(p, 4)
                    ])

                count += 1

    stats_cols = ["Metric", "Adj", "Tech", "N1", "N2", "Median 1", "Median 2", "Stat", "p-value"]
    df = pd.DataFrame(data_results,
                      columns=stats_cols)
    df.to_excel("data/statistical_tests.xlsx", index=False)

    for current_metric in ["R2", "RMSE"]:
        df = pd.read_excel("data/statistical_tests.xlsx", sheet_name=0)

        techs = sorted(list(set(df[df["Metric"] == current_metric]["Tech"])))
        adj_list = sorted(list(set(df[df["Metric"] == current_metric]["Adj"])))
        row_titles = stats_cols[3:]

        data = []
        data_simpl = []
        data_p = []
        data_medians = []

        for tech in techs:

            for i_title in range(len(row_titles)):
                row_title = row_titles[i_title]

                if i_title == 0:
                    line = [tech, row_title]
                else:
                    line = ["", row_title]

                if row_title == "Median 1":
                    line_simpl = [tech, row_title]
                else:
                    line_simpl = ["", row_title]

                line_p = [tech, row_title]
                line_med = [tech, row_title]

                for adj in adj_list:
                    filtered_df = df[(df["Metric"] == current_metric) & (df["Adj"] == adj) & (df["Tech"] == tech)]

                    value = list(filtered_df[row_title])[0]
                    if (current_metric == "R2" and row_title in ["Median 1", "Median 2"]) or row_title in ["p-value"]:
                        value = round(value, 2)
                    else:
                        value = round(value, 0)

                    line.append(value)
                    line_p.append(value)
                    line_med.append(value)
                    line_simpl.append(value)

                data.append(line)
                if row_title == "p-value":
                    data_p.append(line_p)
                if row_title.startswith("Median"):
                    data_medians.append(line_med)
                if row_title in ["Median 1", "Median 2", "p-value"]:
                    data_simpl.append(line_simpl)

        cols = ["Tech", "Stat"]
        cols.extend(adj_list)
        pivot_df = pd.DataFrame(data, columns=cols)
        pivot_df.to_excel("data/stat_pivot_@.xlsx".replace("@", current_metric), index=False)
        pivot_df = pd.DataFrame(data_p, columns=cols)
        pivot_df.to_excel("data/stat_pivot_p-value_@.xlsx".replace("@", current_metric), index=False)
        pivot_df = pd.DataFrame(data_simpl, columns=cols)
        pivot_df.to_excel("data/stat_pivot_simpl_@.xlsx".replace("@", current_metric), index=False)

        pivot_df = pd.DataFrame(data_medians, columns=cols)

        data_medians = []
        for tech in techs:
            line = [tech]
            for adj in adj_list:
                med1 = list(pivot_df[(pivot_df["Tech"] == tech) & (pivot_df["Stat"] == "Median 1")][adj])[0]
                med2 = list(pivot_df[(pivot_df["Tech"] == tech) & (pivot_df["Stat"] == "Median 2")][adj])[0]

                med1 = DICT_FORMAT_METRIC[current_metric].format(float(med1))
                med2 = DICT_FORMAT_METRIC[current_metric].format(float(med2))
                line.append("%s ; %s" % (med1, med2))

            data_medians.append(line)
        cols = ["Tech"]
        cols.extend(adj_list)
        df = pd.DataFrame(data_medians, columns=cols)
        df.to_excel("data/stat_final_medians_@.xlsx".replace("@", current_metric), index=False)
    t2 = time.time()
    print("%d tests executed in %.3f seconds" % (count, t2 - t1))

    ##########################################################
    # Different table requires different code.

    data_results = []
    df_binary_table = pd.read_excel("data/paper/final_analysis/elapsed_time_combinations.xlsx")
    independent_cols = df_binary_table.columns[:7]
    dependent_cols_all = sorted(df_binary_table.columns[7:])
    count = 0

    t1 = time.time()

    for metric in metrics:
        dependent_cols = [col for col in dependent_cols_all if str(col).startswith(metric)]

        for indep_col in independent_cols:
            for dep_col in dependent_cols:
                # print("-" * 50)
                # print(indep_col, dep_col)

                x1, x2 = get_data_for_test(df_binary_table, indep_col, dep_col)

                # print(list(x1))
                # print(list(x2))
                n1 = len(x1)
                n2 = len(x2)
                mean1 = np.median(x1)
                mean2 = np.median(x2)
                stat1, p = stats.mannwhitneyu(x1, x2, alternative=alternative[metric])
                stat2 = n1 * n2 - stat1
                z_score = (stat1 - ((n1 * n2) / 2)) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)

                dep_col = dep_col.split()[-1].strip()
                data_results.append(
                    [
                        metric,
                        DICT_ATTR_NAME[indep_col],
                        DICT_TECH_NAME[dep_col],
                        n1, n2,
                        mean1, mean2,
                        stat1,
                        round(p, 4)
                    ])

                count += 1

        data_medians = []
        for tech in techs:
            line = [tech]
            for adj in adj_list:
                med1 = list(pivot_df[(pivot_df["Tech"] == tech) & (pivot_df["Stat"] == "Median 1")][adj])[0]
                med2 = list(pivot_df[(pivot_df["Tech"] == tech) & (pivot_df["Stat"] == "Median 2")][adj])[0]

                med1 = DICT_FORMAT_METRIC[current_metric].format(float(med1))
                med2 = DICT_FORMAT_METRIC[current_metric].format(float(med2))
                line.append("%s ; %s" % (med1, med2))

            data_medians.append(line)
        cols = ["Tech"]
        cols.extend(adj_list)
        df = pd.DataFrame(data_medians, columns=cols)
        df.to_excel("data/stat_final_medians_@.xlsx".replace("@", current_metric), index=False)
    t2 = time.time()
    print("%d tests executed in %.3f seconds" % (count, t2 - t1))
    stats_cols = ["Metric", "Attr", "Tech", "N1", "N2", "Median 1", "Median 2", "Stat", "p-value"]
    df = pd.DataFrame(data_results,
                      columns=stats_cols)
    df.to_excel("data/statistical_tests_time.xlsx", index=False)


if __name__ == "__main__":
    main()
