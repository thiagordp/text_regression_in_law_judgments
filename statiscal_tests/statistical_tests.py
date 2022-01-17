"""
Statistical Tests for Regression results

"""
import time

import pandas as pd
from scipy import stats


# TODO: get the full results instead of the mean

def get_data_for_test(df_data=pd.DataFrame(), attr_col_name=str(), result_col_name=str()):
    aux = df_data[df_data[attr_col_name] == 0]
    x = aux[result_col_name]

    aux = df_data[df_data[attr_col_name] == 1]
    y = aux[result_col_name]

    return x, y


def main():
    # Load dataset
    metrics = ["MAE", "R2", "RMSE"]
    alternative = {
        "MAE": "greater",
        "R2": "less",
        "RMSE": "greater",
    }

    data_results = []
    df_binary_table = pd.read_excel("data/paper/final_analysis/binary_table.xlsx")
    independent_cols = df_binary_table.columns[:7]
    dependent_cols_all = sorted(df_binary_table.columns[7:])
    count = 0

    t1 = time.time()

    for metric in metrics:
        dependent_cols = [col for col in dependent_cols_all if str(col).startswith(metric)]
        # print("depentent columns:\n%s" % list(dependent_cols))

        for indep_col in independent_cols:
            for dep_col in dependent_cols:
                x1, x2 = get_data_for_test(df_binary_table, indep_col, dep_col)
                stat, p = stats.mannwhitneyu(x1, x2, alternative=alternative[metric])

                dep_col = dep_col.split()[-1].strip()
                data_results.append([metric, indep_col, dep_col, stat, round(p, 4)])

                count += 1
    t2 = time.time()
    print("%d tests executed in %.3f seconds" % (count, t2 - t1))
    df = pd.DataFrame(data_results, columns=["Metric", "Adj", "Tech", "Stat", "p-value"])
    df.to_excel("data/statistical_tests.xlsx", index=False)


if __name__ == "__main__":
    main()
