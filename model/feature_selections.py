"""

@author Thiago R. Dal Pont
"""
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression


def remove_outliers(x, y, sents):
    iforest = IsolationForest(contamination=0.1)
    yhat = iforest.fit_predict(x, y)

    mask = yhat != -1
    x_new = [x[i] for i in range(len(x)) if mask[i]]
    sents_new = [sents[i] for i in range(len(x)) if mask[i]]
    y_new = [y[i] for i in range(len(x)) if mask[i]]

    del x, y, sents, yhat, iforest

    return x_new, y_new, sents_new


def bow_feature_selection(bow, y, k):
    # variance_sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
    # bow = variance_sel.fit_transform(bow)

    test = SelectKBest(score_func=f_regression, k=k)
    fit_bow = test.fit_transform(bow, y)
    new_bow = [list(row) for row in fit_bow]
    del fit_bow

    return new_bow
