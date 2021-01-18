"""

@author Thiago R. Dal Pont
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.preprocessing import MinMaxScaler


def remove_outliers_iforest(x, y, sents):
    iforest = IsolationForest(n_jobs=8, contamination=0.1)
    yhat = iforest.fit_predict(x, y)

    mask = yhat != -1
    x_new = [x[i] for i in range(len(x)) if mask[i]]
    sents_new = [sents[i] for i in range(len(x)) if mask[i]]
    y_new = [y[i] for i in range(len(x)) if mask[i]]

    del x, y, sents, yhat, iforest

    return x_new, y_new, sents_new


def bow_feature_selection(bow, y, k, features_names=[]):
    # variance_sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
    # bow = variance_sel.fit_transform(bow)
    print("Feature Selection....")

    test = SelectKBest(score_func=mutual_info_regression, k=k)
    # test = SelectKBest(score_func=f_regression, k=k)
    fit_bow = test.fit_transform(bow, y)

    new_bow = [list(row) for row in fit_bow]
    del fit_bow

    ################################################

    # ranks = {}
    # ranks["f_reg"] = rank_to_dict(test.scores_, features_names)
    #
    # print("=", end="")

    # lr = LinearRegression(normalize=True)
    # lr.fit(bow, y)
    # ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), features_names)
    # print("=", end="")

    # ridge = Ridge(alpha=7)
    # ridge.fit(bow, y)
    # ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), features_names)
    # print("=", end="")

    # lasso = Lasso(alpha=.05)
    # lasso.fit(bow, y)
    # ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), features_names)
    # print("=", end="")

    # rlasso = RandomizedLasso(alpha=0.04)
    # rlasso.fit(bow, y)
    # ranks["Stability"] = rank_to_dict(np.abs(rlasso.coef_), features_names)
    # print("=", end="")

    # # stop the search when 5 features are left (they will get equal scores)
    # rfe = RFE(rlasso, n_features_to_select=500, verbose=1, step=500)
    # rfe.fit(bow, y)
    #
    # ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), features_names, order=-1)
    # print("=", end="")
    #
    # rf = RandomForestRegressor(max_depth=10)
    # rf.fit(bow, y)
    # ranks["RF"] = rank_to_dict(rf.feature_importances_, features_names)
    # print("=", end="")

    # estimator = RandomizedLasso(weakness=0.2)
    # selector = StabilitySelection(base_estimator=estimator, lambda_name='alpha', threshold=0.9, verbose=1, n_jobs=4)
    # selector.fit(bow, y)
    # ranks["sel"] = rank_to_dict(selector.stability_scores_, features_names)
    # print("=", end="")

    # fig, ax = plot_stability_path(selector)
    # fig.show()

    # r = {}
    # names = {}
    # for name in features_names:
    #     r[name] = round(float(np.mean([ranks[method][name] for method in ranks.keys()])), 2)
    #     names[name] = name
    #
    # methods = sorted(ranks.keys())
    # ranks["Mean"] = r
    #
    # methods.append("Mean")
    # methods.append("Token")
    #
    # ranks["token"] = names

    # print("\t%s" % "\t".join(methods))
    # for name in features_names:
    #     print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))

    # df = pd.DataFrame(data=ranks)
    # df.to_excel("data/test_fs.xlsx", index=False)

    return new_bow


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))
