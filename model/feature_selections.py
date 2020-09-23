"""

@author Thiago R. Dal Pont
"""
from sklearn.feature_selection import SelectKBest, f_regression


def bow_feature_selection(bow, y, k):
    # variance_sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
    # bow = variance_sel.fit_transform(bow)

    test = SelectKBest(score_func=f_regression, k=k)
    bow = test.fit_transform(bow, y)

    return bow
