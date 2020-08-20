"""

@author Thiago Raulino Dal Pont
"""
import random
import time

import numpy as np
import pandas as pd
import tqdm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from util.aux_function import print_time
from visualization.vsm_models_visualization import visualize_decision_tree

REGRESSION_MODELS = {
    "decision_tree": DecisionTreeRegressor(),
    "linear_regression": LinearRegression(),
    "svr_linear": SVR(C=1.0, epsilon=0.2, kernel="linear"),
    "svr_poly_2": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=2),
    "svr_poly_3": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=3),
    "svr_poly_4": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=4),
    "svr_poly_10": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=10),
    "svr_poly_15": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=15),
    "svr_rbf": SVR(C=1.0, epsilon=0.2, kernel="rbf"),
    "svr_sigmoid": SVR(C=1.0, epsilon=0.2, kernel="sigmoid"),
    "random_forest_10": RandomForestRegressor(n_estimators=10),
    "random_forest_50": RandomForestRegressor(n_estimators=50),
    "random_forest_100": RandomForestRegressor(n_estimators=100),
    "random_forest_1000": RandomForestRegressor(n_estimators=1000),
    "random_forest_2000": RandomForestRegressor(n_estimators=2000),
    "random_forest_4000": RandomForestRegressor(n_estimators=4000),
    "random_forest_5000": RandomForestRegressor(n_estimators=5000),
    "mlp_100": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    "mlp_200": MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000),
    "mlp_1000": MLPRegressor(hidden_layer_sizes=(1000,), max_iter=1000),
    "mlp_200_50": MLPRegressor(hidden_layer_sizes=(200, 50,), max_iter=1000),
    "mlp_200_100": MLPRegressor(hidden_layer_sizes=(200, 100,), max_iter=1000),
    "mlp_1000_100": MLPRegressor(hidden_layer_sizes=(1000, 100,), max_iter=1000),
    "mlp_1000_100_50": MLPRegressor(hidden_layer_sizes=(1000, 100, 50,), max_iter=1000),
    "adaboost": AdaBoostRegressor(),
    "bagging": BaggingRegressor(),
    "extra_trees": ExtraTreesRegressor(),
    "gradient_boosting": GradientBoostingRegressor()
}


def full_models_regression(x_train, y_train, x_test, y_test, feature_names):
    results = list()
    print("Training Regressors")
    time.sleep(0.5)

    get_tree = False
    for key in REGRESSION_MODELS.keys():
        print("Training", key)
        print_time()
        time.sleep(1)
        for i in tqdm.tqdm(range(20)):
            model = REGRESSION_MODELS[key]
            model.random_state = random.randint(1, 2 ** 32 - 1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            results.append([key, y_test, y_pred])

            if key.find("decision_tree") != -1 and not get_tree:
                get_tree = True
                visualize_decision_tree(model, feature_names)

                importances = model.feature_importances_
                print(len(importances), len(feature_names))
                indices = np.argsort(importances)[::-1]

                columns = ["feature_name", "importance"]
                list_importances = list()
                for i in range(len(feature_names)):
                    list_importances.append([feature_names[indices[i]], importances[indices[i]]])

                df = pd.DataFrame(data=list_importances, columns=columns)
                df.to_csv("data/features/decision_tree.csv")

    return results


def linear_regression(x_train, y_train, x_test, y_test):
    print("Regressor")
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred


def svm_regression(x_train, y_train, x_test, y_test):
    print("SVR")

    model = SVR(C=1.0, epsilon=0.2)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred


def random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred


def neural_network(x_train, y_train, x_test, y_test):
    model = MLPRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred


def decision_tree(x_train, y_train, x_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred
