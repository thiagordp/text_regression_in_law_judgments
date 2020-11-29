"""

@author Thiago Raulino Dal Pont
"""
import random
import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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
    # "ridge": Ridge(),
    # "sgd_regressor": SGDRegressor(),
    # "random_forest_100_04_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=4, max_leaf_nodes=50,
    # random_state=int(time.time_ns()) % 32),
    # "random_forest_100_5_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=5, max_leaf_nodes=50),
    # "random_forest_100_06_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=6, max_leaf_nodes=50,
    # random_state=int(random.random() * time.time_ns()) % 32),
    # "random_forest_100_7_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=7, max_leaf_nodes=50),
    # "random_forest_100_08_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=8, max_leaf_nodes=50,
    # random_state=int(random.random() * time.time_ns()) % 32),
    # "random_forest_100_9_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=9, max_leaf_nodes=50),
    "random_forest_100_10_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=10, max_leaf_nodes=50,
                                                     random_state=int(time.time_ns()) % 32),
    # "huber": HuberRegressor(),
    # "ransac": RANSACRegressor(),
    # "theil_sen": TheilSenRegressor(),
    # "passive_agressive": PassiveAggressiveRegressor(),
    # "random_forest_1000": RandomForestRegressor(n_estimators=1000, n_jobs=8, max_depth=4, max_leaf_nodes=50),
    # "random_forest_5000": RandomForestRegressor(n_estimators=5000, n_jobs=8, max_depth=4, max_leaf_nodes=50),
    "mlp_200_100": MLPRegressor(hidden_layer_sizes=(200, 100,),
                                max_iter=200,
                                validation_fraction=0.2,
                                early_stopping=True,
                                activation="relu"),
    "mlp_100": MLPRegressor(hidden_layer_sizes=(100,),
                            max_iter=200,
                            validation_fraction=0.2,
                            early_stopping=True,
                            activation="relu"),
    "mlp_200": MLPRegressor(hidden_layer_sizes=(200,),
                            max_iter=200,
                            validation_fraction=0.2,
                            early_stopping=True,
                            activation="relu"),
    "mlp_200_100_50": MLPRegressor(hidden_layer_sizes=(200, 100, 50,),
                                   max_iter=200,
                                   validation_fraction=0.2,
                                   early_stopping=True,
                                   shuffle=True,
                                   activation="relu"),
    # "mlp_100_50_25": MLPRegressor(hidden_layer_sizes=(100, 50, 25,),
    #                               max_iter=200,
    #                               validation_fraction=0.2,
    #                               early_stopping=True,
    #                               shuffle=True,
    #                               activation="relu"),
    "mlp_400_200_100_50": MLPRegressor(hidden_layer_sizes=(400, 200, 100, 50,),
                                       max_iter=200,
                                       validation_fraction=0.2,
                                       early_stopping=True,
                                       shuffle=True,
                                       activation="relu"),
    "mlp_400_200_100": MLPRegressor(hidden_layer_sizes=(400, 200, 100,),
                                    max_iter=200,
                                    validation_fraction=0.2,
                                    early_stopping=True,
                                    shuffle=True,
                                    activation="relu"),

    # "mlp_1000_100": MLPRegressor(hidden_layer_sizes=(1000, 100,),
    #                              max_iter=200,
    #                              validation_fraction=0.2,
    #                              early_stopping=True,
    #                              activation="relu"),
    "adaboost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
    "bagging": BaggingRegressor(n_estimators=100, n_jobs=8, oob_score=True),
    "gradient_boosting": GradientBoostingRegressor(),
    "xgboost": xgb.XGBRegressor(),
    "xgboost_rf": xgb.XGBRFRegressor(),
}

REGRESSION_MODELS["ensemble_voting_bg_mlp_gd"] = VotingRegressor(n_jobs=8, estimators=[
    ('bagging', REGRESSION_MODELS["bagging"]),
    ('mlp', REGRESSION_MODELS["mlp_400_200_100_50"]),
    ('gd', REGRESSION_MODELS["gradient_boosting"])
])

REGRESSION_MODELS["ensemble_voting_en_mlp_mlp"] = VotingRegressor(n_jobs=8, estimators=[
    ('en', REGRESSION_MODELS["elastic_net"]),
    ('mlp', REGRESSION_MODELS["mlp_400_200_100_50"]),
    ('mlp2', REGRESSION_MODELS["mlp_400_200_100"])
])

REGRESSION_MODELS["adaboost_mlp_200"] = AdaBoostRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                          base_estimator=REGRESSION_MODELS["mlp_200"])

REGRESSION_MODELS["adaboost_mlp_100"] = AdaBoostRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                          base_estimator=REGRESSION_MODELS["mlp_100"])

REGRESSION_MODELS["adaboost_mlp_200_100"] = AdaBoostRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                              base_estimator=REGRESSION_MODELS["mlp_200_100"])

REGRESSION_MODELS["adaboost_mlp_200_100_50"] = AdaBoostRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                                 base_estimator=REGRESSION_MODELS["mlp_200_100_50"])

REGRESSION_MODELS["bagging_mlp_200"] = BaggingRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                        base_estimator=REGRESSION_MODELS["mlp_200"],
                                                        n_jobs=8)

REGRESSION_MODELS["bagging_mlp_100"] = BaggingRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                        base_estimator=REGRESSION_MODELS["mlp_100"],
                                                        n_jobs=8)

REGRESSION_MODELS["bagging_mlp_200_100"] = BaggingRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                            base_estimator=REGRESSION_MODELS["mlp_200_100"],
                                                            n_jobs=8)

REGRESSION_MODELS["bagging_mlp_200_100_50"] = BaggingRegressor(random_state=int(time.time()) % (2 ** 32) - 1,
                                                               base_estimator=REGRESSION_MODELS["mlp_200_100_50"],
                                                               n_jobs=8)

#######################################################################################################################

REGRESSION_MODELS_PAPER = {
    "elastic_net": ElasticNet(max_iter=200),
    "ridge": Ridge(max_iter=200),
    "decision_tree": DecisionTreeRegressor(max_depth=10, max_leaf_nodes=100),
    "adaboost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
    "bagging": BaggingRegressor(n_estimators=100, n_jobs=8, oob_score=True),
    "gradient_boosting": GradientBoostingRegressor(max_depth=10, max_leaf_nodes=100),
    "xgboost": xgb.XGBRegressor(),
    "xgboost_rf": xgb.XGBRFRegressor(),
    "mlp_400_200_100_50": MLPRegressor(hidden_layer_sizes=(400, 200, 100, 50,),
                                       max_iter=200,
                                       validation_fraction=0.2,
                                       early_stopping=True,
                                       shuffle=True,
                                       activation="relu"),
    "mlp_400_200_100": MLPRegressor(hidden_layer_sizes=(400, 200, 100,),
                                    max_iter=200,
                                    validation_fraction=0.2,
                                    early_stopping=True,
                                    shuffle=True,
                                    activation="relu"),
    "random_forest": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=10, max_leaf_nodes=100),
    "svr_linear": SVR(C=1.0, epsilon=0.2, kernel="linear", max_iter=200),
    "svr_poly_rbf": SVR(C=1.0, epsilon=0.2, kernel="rbf", max_iter=200)
}

REGRESSION_MODELS_PAPER["ensemble_voting_bg_mlp_gd_xgb"] = VotingRegressor(n_jobs=8, estimators=[
    ('bagging', REGRESSION_MODELS_PAPER["bagging"]),
    ('mlp', REGRESSION_MODELS_PAPER["mlp_400_200_100"]),
    ('xgb', REGRESSION_MODELS_PAPER["xgboost"]),
    ('gd', REGRESSION_MODELS_PAPER["gradient_boosting"])
])

REGRESSION_BIG_MODELS_PAPER = {
    "elastic_net": ElasticNet(),
    "ridge": Ridge(),
    "decision_tree": DecisionTreeRegressor(),
    "adaboost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
    "bagging": BaggingRegressor(n_estimators=100, n_jobs=8),
    "gradient_boosting": GradientBoostingRegressor(),
    "xgboost": xgb.XGBRegressor(),
    "xgboost_rf": xgb.XGBRFRegressor(),
    "mlp_400_200_100_50": MLPRegressor(hidden_layer_sizes=(400, 200, 100, 50,),
                                       max_iter=200,
                                       validation_fraction=0.2,
                                       shuffle=True,
                                       activation="relu"),
    "mlp_400_200_100": MLPRegressor(hidden_layer_sizes=(400, 200, 100,),
                                    max_iter=200,
                                    validation_fraction=0.2,
                                    shuffle=True,
                                    activation="relu"),
    "random_forest_100": RandomForestRegressor(n_estimators=100, n_jobs=8),
    "svr_linear": SVR(C=1.0, epsilon=0.2, kernel="linear"),
    "svr_poly_rbf": SVR(C=1.0, epsilon=0.2, kernel="rbf")
}

REGRESSION_BIG_MODELS_PAPER["ensemble_voting_bg_mlp_gd_xgb"] = VotingRegressor(n_jobs=8, estimators=[
    ('bagging', REGRESSION_BIG_MODELS_PAPER["bagging"]),
    ('mlp', REGRESSION_BIG_MODELS_PAPER["mlp_400_200_100"]),
    ('xgb', REGRESSION_BIG_MODELS_PAPER["xgboost"]),
    ('gd', REGRESSION_BIG_MODELS_PAPER["gradient_boosting"])
])


def full_models_regression(x_train, y_train, x_test, y_test, feature_names, tech_representation, papers_models=False, reduce_models=True):
    results_test = list()
    results_train = list()
    # print("Training Regressors")

    time.sleep(0.5)

    # Choose the dictionary of models
    if papers_models:

        #############################################
        #              Reduce Models                #
        #############################################
        if reduce_models:
            models_list = REGRESSION_MODELS_PAPER
        else:
            models_list = REGRESSION_BIG_MODELS_PAPER
    else:
        models_list = REGRESSION_MODELS

    # For each model, fit it to the data and make predictions
    for key in models_list.keys():

        # TODO: Remove after making predictions
        # if key != "ensemble_voting_bg_mlp_gd_xgb":
        #     continue

        # print("Training", key)
        model = models_list[key]

        model.random_state = int(random.random() * time.time() * random.randint(1, 2 ** 32 - 1)) % 2 ** 32
        model.fit(np.array(x_train), np.array(y_train))

        y_pred = model.predict(np.array(x_test))
        results_test.append([key, y_test, y_pred])

        y_pred = model.predict(np.array(x_train))
        results_train.append([key, y_train, y_pred])

        # if i == 0:
        #     vsm_models_visualization.setup_visualization(key, model, feature_names, tech_representation)

    return results_train, results_test
