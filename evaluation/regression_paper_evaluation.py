"""
Evaluate result logs from paper experiments

@author Thiago Dal Pont
@date Jan 28th, 2021
"""
import glob
import json
import re
import string
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def paper_results_evaluation():
    print("=" * 128)
    print("Paper Results Evaluation")
    print("-" * 128)

    results_logs = [log_result for log_result in glob.glob("data/paper/final_results/*.csv") if log_result.find("_table") == -1]

    print("Available Combinations:", len(results_logs))

    ##########################    SECTION 5.1: BASELINE (BL) AND FULL PIPELINE (FP)    ##########################
    log_baseline = "data/paper/final_results/results_regression_wo_fs_tf_w_or1_wo_ng_wo_at_wo_cv_wo_oa_wo_or2.csv"
    log_full_pipeline = "data/paper/final_results/results_regression_w_fs_before_500_tf_w_or1_w_ng_w_at_w_cv_w_oa_w_or2.csv"

    # df_baseline = pd.read_csv(log_baseline)
    df_fp = pd.read_csv(log_full_pipeline)
    techs_fp = sorted(set(df_fp["tech"]))
    print("Techs:", techs_fp)

    # Get data from FP results
    rmse_fp = dict()
    r2_fp = dict()
    mae_fp = dict()
    for tech in techs_fp:
        df_tech = df_fp[df_fp["tech"] == tech]
        rmse_fp[tech] = np.mean(df_tech["rmse_test"])
        r2_fp[tech] = round(np.mean(df_tech["r2_test"]), 2)
        mae_fp[tech] = np.mean(df_tech["mae_test"])

    plot_paper_rmse_r2_results(techs_fp, rmse_fp, r2_fp, mae_fp, "data/paper/final_analysis/full_pipeline_r2_rmse.pdf")
    # plot_paper_rmse_r2_results(techs_fp, rmse_fp, r2_fp, mae_fp, "data/paper/final_analysis/baseline_r2_rmse.pdf")

    ##########################    SECTION 5.2: COMBINATIONS RESULTS    ##########################
    columns_combinations = [
        "fs", "or1", "ng", "at", "cv", "oa", "or2",
        "@ adaboost",
        "@ bagging",
        "@ decision_tree",
        "@ elastic_net",
        "@ ensemble_voting_bg_mlp_gd_xgb",
        "@ gradient_boosting",
        "@ mlp",
        "@ random_forest",
        "@ ridge",
        "@ svr_poly_rbf",
        "@ xgboost"
    ]

    r2_columns = [line.replace("@", "R2") for line in columns_combinations]
    rmse_columns = [line.replace("@", "RMSE") for line in columns_combinations]

    # Plot the graphs of the descending metrics
    r2_binary_table_df = pd.read_csv("data/paper/final_analysis/binary_table.csv", usecols=r2_columns)
    r2_binary_table_df["combination"] = r2_binary_table_df["fs"].astype(str) + \
                                        r2_binary_table_df["or1"].astype(str) + \
                                        r2_binary_table_df["ng"].astype(str) + \
                                        r2_binary_table_df["at"].astype(str) + \
                                        r2_binary_table_df["cv"].astype(str) + \
                                        r2_binary_table_df["oa"].astype(str) + \
                                        r2_binary_table_df["or2"].astype(str)

    rmse_binary_table_df = pd.read_csv("data/paper/final_analysis/binary_table.csv", usecols=rmse_columns)
    rmse_binary_table_df["combination"] = rmse_binary_table_df["fs"].astype(str) + \
                                          rmse_binary_table_df["or1"].astype(str) + \
                                          rmse_binary_table_df["ng"].astype(str) + \
                                          rmse_binary_table_df["at"].astype(str) + \
                                          rmse_binary_table_df["cv"].astype(str) + \
                                          rmse_binary_table_df["oa"].astype(str) + \
                                          rmse_binary_table_df["or2"].astype(str)

    plot_paper_combinations_results(r2_binary_table_df, rmse_binary_table_df)

    ##########################    SECTION 5.3: IMPACT OF EACH ADJUSTMENT    ##########################

    r2_binary_table_df = pd.read_csv("data/paper/final_analysis/binary_table.csv", usecols=r2_columns)
    r2_binary_table_df["combination"] = r2_binary_table_df["fs"].astype(str) + \
                                        r2_binary_table_df["or1"].astype(str) + \
                                        r2_binary_table_df["ng"].astype(str) + \
                                        r2_binary_table_df["at"].astype(str) + \
                                        r2_binary_table_df["cv"].astype(str) + \
                                        r2_binary_table_df["oa"].astype(str) + \
                                        r2_binary_table_df["or2"].astype(str)

    rmse_binary_table_df = pd.read_csv("data/paper/final_analysis/binary_table.csv", usecols=rmse_columns)
    rmse_binary_table_df["combination"] = rmse_binary_table_df["fs"].astype(str) + \
                                          rmse_binary_table_df["or1"].astype(str) + \
                                          rmse_binary_table_df["ng"].astype(str) + \
                                          rmse_binary_table_df["at"].astype(str) + \
                                          rmse_binary_table_df["cv"].astype(str) + \
                                          rmse_binary_table_df["oa"].astype(str) + \
                                          rmse_binary_table_df["or2"].astype(str)

    extract_execution_time()

    execution_time_df = pd.read_csv("data/paper/final_analysis/elapsed_time_combinations.csv")
    execution_time_df["combination"] = execution_time_df["fs"].astype(str) + \
                                       execution_time_df["or1"].astype(str) + \
                                       execution_time_df["ng"].astype(str) + \
                                       execution_time_df["at"].astype(str) + \
                                       execution_time_df["cv"].astype(str) + \
                                       execution_time_df["oa"].astype(str) + \
                                       execution_time_df["or2"].astype(str)

    table_paper_adjustments_impact(r2_binary_table_df, "R2")
    table_paper_adjustments_impact(rmse_binary_table_df, "RMSE")
    table_paper_adjustments_impact(execution_time_df, "Time")


def extract_execution_time():
    print("-" * 16, "Extract Execution Time", "-" * 16)
    path_run_logs = "data/paper/running_logs/*.log"

    log_files = glob.glob(path_run_logs)

    # print(log_files)
    full_text = ""
    for log_path in log_files:
        full_text += open(log_path).read() + "\n"

    full_text = "".join(filter(lambda char: char in string.printable, full_text))

    n_experiments = 0
    experiment_logs = list(full_text.split("PAPER EXPERIMENTS"))

    results_list = []
    for line in experiment_logs:

        if line.find("Tech:") < 0:
            continue

        line = line.replace("True", "1").replace("False", "0").replace("\t", "")

        # Feature selection extraction
        fs = find_adjustment_setup(line, r'Feature Selection:\s+\d')
        cv = find_adjustment_setup(line, r'Cross Validation:\s+\d')
        or1 = find_adjustment_setup(line, r'Remove Outliers:\s+\d')
        at = find_adjustment_setup(line, r'Include Attributes:\s+\d')
        ng = find_adjustment_setup(line, r'Include N-Grams:\s+\d')
        oa = find_adjustment_setup(line, r'Reduce Models:\s+\d')
        or2 = find_adjustment_setup(line, r'Remove outlier_both\s+\d')

        elapsed_time = find_adjustment_setup(line, r"Elapsed Time:\s+\d+:\d+:\d+.\d+", True)

        if elapsed_time is None:
            # Elapsed Time: 1 day, 11:09:35.442229
            elapsed_time = find_adjustment_setup(line, r"Elapsed Time:\s+\d+\s+day,\s+\d+:\d+:\d+.\d+", True)

        result_line = [fs, cv, or1, at, ng, oa, or2, elapsed_time]
        results_list.append(result_line)

    results_df = pd.DataFrame(results_list, columns=["fs", "cv", "or1", "at", "ng", "oa", "or2", "elapsed_time"])

    file_output = "data/paper/final_analysis/elapsed_time_combinations.@"

    results_df.to_csv(file_output.replace("@", "csv"), index=False)
    results_df.to_excel(file_output.replace("@", "xlsx"), index=False)
    results_df.corr().to_excel(file_output.replace("@", "corr.xlsx"))


def find_adjustment_setup(text, re_expression, multiple_matches=False):
    match_string = re.findall(re_expression, text)

    if len(match_string) > 1 and not multiple_matches:
        print("Multiple Matches")
        return None

    if len(match_string) > 0 and not multiple_matches:
        res = [int(i) for i in match_string[0].split() if i.isdigit()]
        return res[0]

    if len(match_string) > 0 and multiple_matches:
        split_string = match_string[0].replace(".", ":").replace(",", ":")
        res = [int(str(i).strip()) for i in split_string.split(":") if str(i).strip().isdigit()]

        if match_string[0].find("day") >= 0:
            # There is no log with more than 1 day and X hours
            hours = 24 + res[0] + res[1] / 60 + res[2] / 3600
        else:
            if len(res) != 4:
                print("Ops")

            hours = res[0] + res[1] / 60 + res[2] / 3600

        return hours

    return None


def table_paper_adjustments_impact(table_df, metric):
    print("-" * 20, "Table Adjustments Impact - ", metric, "-" * 20)

    list_adjustments = ["fs", "at", "cv", "ng", "oa", "or1", "or2"]

    # OR1 and OR2 is separately
    if metric != "Time":
        columns_combinations = [
            "@ adaboost",
            "@ bagging",
            "@ decision_tree",
            "@ elastic_net",
            "@ ensemble_voting_bg_mlp_gd_xgb",
            "@ gradient_boosting",
            "@ mlp",
            "@ random_forest",
            "@ ridge",
            "@ svr_poly_rbf",
            "@ xgboost"
        ]
    else:
        columns_combinations = [
            "elapsed_time"
        ]

    ###### R2 #####
    metric_combinations = [text.replace("@", metric) for text in columns_combinations]

    dict_diff = dict()
    dict_count = dict()

    for adjust in list_adjustments:
        print("." * 30, "Adjustment: ", adjust, "." * 30)

        results_df_zero = table_df.loc[(table_df[adjust] == 0)]
        results_df_one = table_df.loc[(table_df[adjust] == 1)]

        if adjust == "or1":
            results_df_zero = results_df_zero.loc[(results_df_zero["or2"] == 0)]
        if adjust == "or2":
            results_df_zero = results_df_zero.loc[(results_df_zero["or1"] == 0)]

        for index, row in results_df_zero.iterrows():

            match_df = results_df_one.copy()
            if adjust != "fs":
                match_df = match_df.loc[(match_df["fs"] == int(row["fs"]))]
            if adjust != "ng":
                match_df = match_df.loc[(match_df["ng"] == int(row["ng"]))]
            if adjust != "at":
                match_df = match_df.loc[(match_df["at"] == int(row["at"]))]
            if adjust != "cv":
                match_df = match_df.loc[(match_df["cv"] == int(row["cv"]))]
            if adjust != "oa":
                match_df = match_df.loc[(match_df["oa"] == int(row["oa"]))]
            if adjust != "or2":
                match_df = match_df.loc[(match_df["or2"] == int(row["or2"]))]
            if adjust != "or1":
                match_df = match_df.loc[(match_df["or1"] == int(row["or1"]))]

            compare_row = 1
            for index, matchx in match_df.iterrows():
                compare_row = matchx

            if type(compare_row) is int:
                print("Skip", row["fs"], row["or1"], row["ng"], row["at"], row["cv"], row["oa"], row["or2"])
                continue

            for tech_result in metric_combinations:

                metric_zero = row[tech_result]
                metric_one = compare_row[tech_result]

                if metric != "Time":
                    diff_metric = metric_one - metric_zero
                else:
                    diff_metric = 100 * ((metric_one / metric_zero) - 1)

                # Update to append to array not replace.

                if adjust in dict_diff.keys():
                    dict_tech = dict_diff[adjust]

                    if tech_result in dict_tech.keys():
                        list_results = dict_tech[tech_result]
                        list_results.append(diff_metric)
                        dict_tech[tech_result] = list_results

                    else:
                        dict_tech[tech_result] = [diff_metric]
                else:
                    dict_tech = dict()
                    dict_tech[tech_result] = [diff_metric]

                    dict_diff[adjust] = dict_tech

    for adj in dict_diff.keys():
        dict_tech = dict_diff[adj]
        for tech_result in dict_tech.keys():
            if metric == "R2":

                # print("Median:", round(np.median(dict_tech[tech_result]), 4))
                # print("Avg:", round(np.mean(dict_tech[tech_result]), 4))
                # print("Std:", round(np.std(dict_tech[tech_result]), 4))
                # print("Var:", round(np.var(dict_tech[tech_result]), 4))
                # print("Err:", round(stats.sem(dict_tech[tech_result]), 4))
                # print("Desc:", stats.describe(dict_tech[tech_result]))
                # print("Entropy:", round(stats.entropy(dict_tech[tech_result]), 4))

                str_metric = str(round(np.mean(dict_tech[tech_result]), 2)).replace(".0 ", ".00 ") + " ± " + str(round(np.std(dict_tech[tech_result]), 2))
                # str_metric = str(round(np.mean(dict_tech[tech_result]), 2)) + " ± " + str(round(np.std(dict_tech[tech_result]),2))

            elif metric == "Time":
                str_metric = str(round(np.mean(dict_tech[tech_result]), 1)) + " ± " + str(round(np.std(dict_tech[tech_result]), 1))
            else:
                str_metric = str(int(round(np.mean(dict_tech[tech_result]), 0))) + " ± " + str(int(round(np.std(dict_tech[tech_result]), 0)))
            dict_tech[tech_result] = str_metric

    output_json_name = "data/paper/final_analysis/results_combinations_" + str(metric).lower() + ".json"

    with open(output_json_name, "w+") as f_out:
        f_out.write(json.dumps(dict_diff, indent=4))

    export_impact_results_to_table(dict_diff, metric)


def export_impact_results_to_table(dict_diff, metric_label=""):
    metrics = list(dict_diff["fs"].keys())

    columns = ["technique"]
    columns.extend(list(dict_diff.keys()))

    lines = []

    for metric in metrics:
        line_values = [metric]
        for adjustment in dict_diff.keys():
            line_values.append(dict_diff[adjustment][metric])
        lines.append(line_values)

    metrics_df = pd.DataFrame(lines, columns=columns)

    file_output_name = "data/paper/final_analysis/adjustments_impact_" + str(metric_label).lower() + ".@"
    metrics_df.to_csv(file_output_name.replace("@", "csv"), index=False)
    metrics_df.to_excel(file_output_name.replace("@", "xlsx"), index=False)


def plot_paper_combinations_results(r2_df, rmse_df):
    r2_df.sort_values(by="R2 ensemble_voting_bg_mlp_gd_xgb", ascending=False, inplace=True)
    rmse_df.sort_values(by="RMSE ensemble_voting_bg_mlp_gd_xgb", ascending=True, inplace=True)

    combinations_r2 = r2_df["combination"]
    combinations_rmse = rmse_df["combination"]

    r2_df.drop(columns=["combination", "fs", "ng", "oa", "or1", "or2", "cv", "at"], inplace=True)
    rmse_df.drop(columns=["combination", "fs", "ng", "oa", "or1", "or2", "cv", "at"], inplace=True)

    columns_key_r2 = sorted(r2_df.columns)
    columns_key_rmse = sorted(rmse_df.columns)
    columns = ["AdaBoost",
               "Bagging",
               "Decision Tree",
               "Elastic Net",
               "Ensemble Voting",
               "Gradient Boosting",
               "Neural Network",
               "Random Forest",
               "Ridge",
               "SVM",
               "XGBoosting"
               ]

    colors_techs = [
        "royalblue",
        "grey",
        "darkcyan",
        "darkkhaki",
        "teal",
        "firebrick",
        "mediumseagreen",
        "chocolate",
        "darkorange",
        "purple",
        "seagreen"
    ]

    colors_real = [
        (215, 38, 61),
        (244, 96, 54),
        (46, 41, 78),
        (27, 153, 139),
        (169, 183, 97),
        (123, 44, 191),
        (238, 150, 75),
        (0, 126, 167),
        (150, 48, 63),
        (119, 191, 163),
        (0, 0, 0),
    ]

    for it_colors in range(len(colors_real)):
        color_real = colors_real[it_colors]
        new_tuple = list()
        for color_ind in color_real:
            color_ind /= 255
            new_tuple.append(color_ind)

        colors_real[it_colors] = tuple(new_tuple)

    line_styles = [
        "solid"
    ]

    dot_styles = [
        "o",
        "v",
        "s",
    ]

    plt.close('all')
    fig, ax = plt.subplots()

    fig.set_figheight(5)
    fig.set_figwidth(9)

    ax.grid(axis="y")

    min_lim, max_lim = -0.4, 0.8

    ax.grid(axis="y", linestyle=":")  # List of Colors available in: https://matplotlib.org/3.1.0/gallery/color/named_colors.html

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    for it_columns in range(len(columns)):
        col_key = columns_key_r2[it_columns]
        col_name = columns[it_columns]
        values = r2_df[col_key]
        c = colors_real[it_columns]
        line_style = line_styles[it_columns % len(line_styles)]

        plt.plot(combinations_r2, values,
                 label=col_name, color=c,
                 linestyle=line_style,
                 ms=2,
                 alpha=0.65,
                 marker=dot_styles[it_columns % len(dot_styles)])
    ax.set_xticks(combinations_r2)
    ax.set_xticklabels(combinations_r2, rotation=90, fontsize=8)
    plt.yticks(fontsize=12)
    ax.set_ylim(min_lim, max_lim)
    ax.set_xlim(np.array([2.5, -2.5]) + ax.get_xlim())
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.set_ylabel('R2', color="darkslategray", fontsize=10)
    ax.set_xlabel('Combinations', color="darkslategray", fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=6)

    plt.subplots_adjust(left=0.0, right=1)
    fig.tight_layout()
    plt.savefig("data/paper/final_analysis/combinations_r2.pdf")

    ################ RMSE PLOT ####################
    plt.close('all')
    fig, ax = plt.subplots()

    fig.set_figheight(5)
    fig.set_figwidth(9)

    ax.grid(axis="y")

    min_lim, max_lim = 1500, 4500

    ax.grid(axis="y", linestyle=":")  # List of Colors available in: https://matplotlib.org/3.1.0/gallery/color/named_colors.html

    for it_columns in range(len(columns)):
        col_key = columns_key_rmse[it_columns]
        col_name = columns[it_columns]
        values = rmse_df[col_key]
        c = colors_real[it_columns]
        line_style = line_styles[it_columns % len(line_styles)]
        plt.plot(combinations_rmse, values,
                 label=col_name, color=c,
                 linestyle=line_style,
                 ms=2,
                 alpha=0.65,
                 marker=dot_styles[it_columns % len(dot_styles)])

    ax.set_xticklabels(combinations_r2, rotation=90, fontsize=8)
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    plt.yticks(fontsize=12)
    ax.set_ylim(min_lim, max_lim)
    ax.set_xlim(np.array([2.5, -2.5]) + ax.get_xlim())
    ax.set_ylabel('RMSE', color="darkslategray", fontsize=10)
    ax.set_xlabel('Combinations', color="darkslategray", fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=6)
    fig.tight_layout()
    plt.savefig("data/paper/final_analysis/combinations_rmse.pdf")


def plot_paper_rmse_r2_results(techs, rmse_dict, r2_dict, mae_dict, output_path):
    tech_names = list()

    tech_names.append("AdaBoost")
    tech_names.append("Bagging")
    tech_names.append("Decision Tree")
    tech_names.append("Elastic Net")
    tech_names.append("Ensemble Voting")
    tech_names.append("Gradient Boosting")
    tech_names.append("Neural Network")
    tech_names.append("Random Forest")
    tech_names.append("Ridge")
    tech_names.append("SVM")
    tech_names.append("XGBoosting")

    tech_names = sorted(set(tech_names))

    rmse_tech_value = list()
    r2_tech_value = list()

    matplotlib.rcParams['font.family'] = "FreeSerif"

    for i_tech in range(len(techs)):
        tech_key = techs[i_tech]

        rmse_tech_value.append(rmse_dict[tech_key])
        r2_tech_value.append(r2_dict[tech_key])

    # RMSE
    fig, ax1 = plt.subplots()

    fig.set_figheight(6)
    fig.set_figwidth(14)
    ax1.set_ylabel('RMSE', color="darkslategray", fontsize=12)
    min_lim, max_lim = get_lim(rmse_tech_value, "RMSE")
    ax1.set_ylim(min_lim, max_lim)

    ax1.grid(axis="y", linestyle=":")  # List of Colors available in: https://matplotlib.org/3.1.0/gallery/color/named_colors.html

    for i in range(len(techs)):
        tech = techs[i]
        tech_name = tech_names[i]
        rmse_value = round(float(rmse_tech_value[i]), 0)

        ax1.bar(tech_name, rmse_value, color="lightsteelblue")
        label = format(rmse_value, ',.0f')

        plt.annotate(label,  # this is the text
                     (tech_name, rmse_value - 700),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 3),  # distance from text to points (x,y)
                     color="darkslategray",
                     fontsize=12,
                     ha='center')  # horizontal alignment can be left, right or center

    ax1.tick_params(axis='y', labelcolor="darkslategray", labelsize=12)
    ax1.set_xticklabels(tech_names, rotation=35, fontsize=12)
    ylabels = [format(label, ',.0f') for label in ax1.get_yticks()]
    ax1.set_yticklabels(ylabels)

    # R2 Plot
    ax2 = ax1.twinx()

    # ax2.grid(axis="y", linestyle=":", color="mediumaquamarine")
    color = 'tab:blue'
    ax2.set_ylabel('R²', color="seagreen", fontsize=12)  # we already handled the x-label with ax1
    ax2.plot(tech_names, r2_tech_value, "-s", color="mediumseagreen")
    ax2.set_xticklabels(tech_names, rotation=35, fontsize=12)

    lim_min, lim_max = get_lim(r2_tech_value, "R2")
    ax2.set_ylim(lim_min, lim_max)
    ax2.tick_params(axis='y', labelcolor="seagreen", labelsize=12)
    ylabels = [format(label, ',.2f') for label in ax2.get_yticks()]
    ax2.set_yticklabels(ylabels)

    for tech, data_tech in zip(tech_names, r2_tech_value):
        label = format(data_tech, ',.2f')

        plt.annotate(label,  # this is the text
                     (tech, data_tech),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     color="darkslategray",
                     fontsize=12,
                     ha='center')  # horizontal alignment can be left, right or center

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_path)


def get_lim(data, type):
    min_data = min(data)
    max_data = max(data)

    if type == "R2":

        delta = 0.2
        min_lim = 0
        max_lim = 1

        while min_lim > min_data:
            min_lim -= delta

        while max_lim < max_data:
            max_lim += delta

        return min_lim, max_lim
    elif type == "RMSE":
        delta = 500
        min_lim = 0
        max_lim = 3000

        while min_lim > min_data:
            min_lim -= delta

        while max_lim < max_data:
            max_lim += delta

        return min_lim, max_lim
    else:
        print("Error type")

    return 0, 1
