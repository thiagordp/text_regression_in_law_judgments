"""

@author Thiago Raulino Dal Pont
"""
import random
import time

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._testing import ignore_warnings

# REGRESSION_MODELS = {
# "random_forest_5": RandomForestRegressor(n_estimators=5, n_jobs=8),
# "decision_tree": DecisionTreeRegressor(),
# "linear_regression": LinearRegression(),
# "svr_linear": SVR(C=1.0, epsilon=0.2, kernel="linear"),
# "svr_poly_2": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=2),
# "svr_poly_3": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=3),
# "svr_poly_4": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=4),
# "svr_poly_10": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=10),
# "svr_poly_15": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=15),
# "svr_rbf": SVR(C=1.0, epsilon=0.2, kernel="rbf"),
# "svr_sigmoid": SVR(C=1.0, epsilon=0.2, kernel="sigmoid"),
# "random_forest_10": RandomForestRegressor(n_estimators=10, n_jobs=8),
# "random_forest_50": RandomForestRegressor(n_estimators=50, n_jobs=8),
# "random_forest_100": RandomForestRegressor(n_estimators=100, n_jobs=8),
# "random_forest_1000": RandomForestRegressor(n_estimators=1000, n_jobs=8),
# "random_forest_2000": RandomForestRegressor(n_estimators=2000, n_jobs=8),
# "random_forest_4000": RandomForestRegressor(n_estimators=4000, n_jobs=8),
# "random_forest_5000": RandomForestRegressor(n_estimators=5000, n_jobs=8),
# "mlp_100": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
# "mlp_200": MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000),
# "mlp_1000": MLPRegressor(hidden_layer_sizes=(1000,), max_iter=1000),
# "mlp_200_50": MLPRegressor(hidden_layer_sizes=(200, 50,), max_iter=1000),
# "mlp_200_100": MLPRegressor(hidden_layer_sizes=(200, 100,), max_iter=1000),
# "mlp_1000_100": MLPRegressor(hidden_layer_sizes=(1000, 100,), max_iter=1000),
# "mlp_1000_100_50": MLPRegressor(hidden_layer_sizes=(1000, 100, 50,), max_iter=1000),
# "adaboost": AdaBoostRegressor(),
# "bagging": BaggingRegressor(),
# "extra_trees": ExtraTreesRegressor(),
# "gradient_boosting": GradientBoostingRegressor()
# }

REGRESSION_MODELS = {
    "decision_tree": DecisionTreeRegressor(max_depth=4, max_leaf_nodes=50),
    # "linear_regression": LinearRegression(n_jobs=8),
    "elastic_net": ElasticNet(),
    "ridge": Ridge(),
    # "sgd_regressor": SGDRegressor(),
    "random_forest_100_4_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=4, max_leaf_nodes=50),
    "random_forest_100_5_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=5, max_leaf_nodes=50),
    "random_forest_100_4_100": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=4, max_leaf_nodes=100),
    "random_forest_100_5_100": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=5, max_leaf_nodes=100),
    # "random_forest_1000": RandomForestRegressor(n_estimators=1000, n_jobs=8, max_depth=4, max_leaf_nodes=50),
    # "random_forest_5000": RandomForestRegressor(n_estimators=5000, n_jobs=8, max_depth=4, max_leaf_nodes=50),
    "mlp_200_100": MLPRegressor(hidden_layer_sizes=(200, 100,),
                                max_iter=200,
                                validation_fraction=0.2,
                                early_stopping=True,
                                activation="relu"),
    # "mlp_100": MLPRegressor(hidden_layer_sizes=(100,),
    #                         max_iter=200,
    #                         validation_fraction=0.2,
    #                         early_stopping=True,
    #                         activation="relu"),
    # "mlp_200": MLPRegressor(hidden_layer_sizes=(200,),
    #                         max_iter=200,
    #                         validation_fraction=0.2,
    #                         early_stopping=True,
    #                         activation="relu"),
    "mlp_200_100_50": MLPRegressor(hidden_layer_sizes=(200, 100, 50,),
                                   max_iter=200,
                                   validation_fraction=0.2,
                                   early_stopping=True,
                                   activation="relu"),
    "mlp_100_50_25": MLPRegressor(hidden_layer_sizes=(100, 50, 25,),
                                  max_iter=200,
                                  validation_fraction=0.2,
                                  early_stopping=True,
                                  activation="relu"),
    "mlp_400_200_100": MLPRegressor(hidden_layer_sizes=(400, 200, 100,),
                                    max_iter=200,
                                    validation_fraction=0.2,
                                    early_stopping=True,
                                    activation="relu"),
    # "mlp_1000_100": MLPRegressor(hidden_layer_sizes=(1000, 100,),
    #                              max_iter=200,
    #                              validation_fraction=0.2,
    #                              early_stopping=True,
    #                              activation="relu"),
    "adaboost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
    "bagging": BaggingRegressor(n_estimators=100, n_jobs=8, oob_score=True),
    "gradient_boosting": GradientBoostingRegressor(random_state=int(time.time()) % (2 ** 32) - 1)
}


# REGRESSION_MODELS = {
#     "ridge": Ridge(),
#     "random_forest_500": RandomForestRegressor(n_estimators=500, n_jobs=8, max_depth=4, max_leaf_nodes=50),
#     "mlp_200_100_50": MLPRegressor(hidden_layer_sizes=(200, 100, 50,),
#                                    max_iter=200,
#                                    validation_fraction=0.2,
#                                    early_stopping=True,
#                                    activation="relu"),
#     "bagging": BaggingRegressor(n_estimators=100, n_jobs=8, oob_score=True),
#     "gradient_boosting": GradientBoostingRegressor(random_state=int(time.time()) % (2 ** 32) - 1)
# }


@ignore_warnings(category=ConvergenceWarning)
def full_models_regression(x_train, y_train, x_test, y_test, feature_names, tech_representation):
    results_test = list()
    results_train = list()
    # print("Training Regressors")
    time.sleep(0.5)

    for key in REGRESSION_MODELS.keys():
        # print("Training", key)
        time.sleep(1)

        for i in range(1):
            model = REGRESSION_MODELS[key]

            model.random_state = random.randint(1, 2 ** 32 - 1)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            results_test.append([key, y_test, y_pred])

            y_pred = model.predict(x_train)
            results_train.append([key, y_train, y_pred])

            # if i == 0:
            #     vsm_models_visualization.setup_visualization(key, model, feature_names, tech_representation)

    return results_train, results_test


def linear_regression(x_train, y_train, x_test, y_test):
    # print("Regressor")
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred


def svm_regression(x_train, y_train, x_test, y_test):
    # print("SVR")

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
