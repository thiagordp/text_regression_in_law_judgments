"""

@author Thiago Raulino Dal Pont
"""
import random
import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
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

REGRESSION_MODELS["emsemble_voting_bg_mlp_gd"] = VotingRegressor(n_jobs=8, estimators=[
    ('bagging', REGRESSION_MODELS["bagging"]),
    ('mlp', REGRESSION_MODELS["mlp_400_200_100_50"]),
    ('gd', REGRESSION_MODELS["gradient_boosting"])
])

REGRESSION_MODELS["emsemble_voting_en_mlp_mlp"] = VotingRegressor(n_jobs=8, estimators=[
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


REGRESSION_MODELS_PAPER = {
    "ridge": Ridge(),
    "decision_tree": DecisionTreeRegressor(max_depth=4, max_leaf_nodes=50),
    "adaboost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
    "bagging": BaggingRegressor(n_estimators=100, n_jobs=8, oob_score=True),
    "gradient_boosting": GradientBoostingRegressor(),
    "xgboost": xgb.XGBRegressor(),
    "xgboost_rf": xgb.XGBRFRegressor(),
    "extra_trees": ExtraTreesRegressor(),
    # "mlp_5": MLPRegressor(hidden_layer_sizes=(5,),
    #                       max_iter=200,
    #                       validation_fraction=0.2,
    #                       early_stopping=True,
    #                       shuffle=True,
    #                       activation="relu"),
    # "mlp_20": MLPRegressor(hidden_layer_sizes=(20,),
    #                        max_iter=200,
    #                        validation_fraction=0.2,
    #                        early_stopping=True,
    #                        shuffle=True,
    #                        activation="relu"),
    # "mlp_400_200_100_50": MLPRegressor(hidden_layer_sizes=(400, 200, 100, 50,),
    #                                    max_iter=200,
    #                                    validation_fraction=0.2,
    #                                    early_stopping=True,
    #                                    shuffle=True,
    #                                    activation="relu"),
    # "mlp_400_200_100": MLPRegressor(hidden_layer_sizes=(400, 200, 100,),
    #                                 max_iter=200,
    #                                 validation_fraction=0.2,
    #                                 early_stopping=True,
    #                                 shuffle=True,
    #                                 activation="relu"),
    # "random_forest_100_10_50": RandomForestRegressor(n_estimators=100, n_jobs=8, max_depth=10, max_leaf_nodes=50),
    # "svr_linear": SVR(C=1.0, epsilon=0.2, kernel="linear"),
    # "svr_poly_2": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=2),
    # "svr_poly_3": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=3),
    # "svr_poly_4": SVR(C=1.0, epsilon=0.2, kernel="poly", degree=4),
    # "svr_poly_rbf": SVR(C=1.0, epsilon=0.2, kernel="rbf"),
}


#
# REGRESSION_MODELS_PAPER["bagging_mlp_5"] = BaggingRegressor(base_estimator=REGRESSION_MODELS_PAPER["mlp_5"], n_jobs=8)
# REGRESSION_MODELS_PAPER["bagging_mlp_20"] = BaggingRegressor(base_estimator=REGRESSION_MODELS_PAPER["mlp_20"], n_jobs=8)
# REGRESSION_MODELS_PAPER["emsemble_voting_bg_mlp_gd"] = VotingRegressor(n_jobs=8, estimators=[
#     ('bagging', REGRESSION_MODELS_PAPER["bagging"]),
#     ('mlp', REGRESSION_MODELS_PAPER["mlp_400_200_100_50"]),
#     ('gd', REGRESSION_MODELS_PAPER["gradient_boosting"])
# ])
#
# estimators = [
#     ('rf', RandomForestRegressor(n_estimators=100, max_depth=4, max_leaf_nodes=50)),
#     ('gb', GradientBoostingRegressor())
# ]
#
# REGRESSION_MODELS_PAPER["stacking_rf_mlp"] = StackingRegressor(
#     estimators=estimators,
#     final_estimator=MLPRegressor(hidden_layer_sizes=(400, 200, 100,),
#                                  max_iter=200,
#                                  validation_fraction=0.2,
#                                  early_stopping=True,
#                                  shuffle=True,
#                                  activation="relu")
# )


# "stacking": StackingRegressor(estimators=[
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('svr', MLPRegressor))
# ], final_estimator = LogisticRegression() )


def full_models_regression(x_train, y_train, x_test, y_test, feature_names, tech_representation, papers_models=False):
    results_test = list()
    results_train = list()
    # print("Training Regressors")
    time.sleep(0.5)

    if papers_models:
        models_list = REGRESSION_MODELS_PAPER
    else:
        models_list = REGRESSION_MODELS

    for key in models_list.keys():
        # print("Training", key)
        time.sleep(1)

        for i in range(1):
            model = models_list[key]

            model.random_state = random.randint(1, 2 ** 32 - 1)
            model.fit(np.array(x_train), np.array(y_train))

            y_pred = model.predict(np.array(x_test))
            results_test.append([key, y_test, y_pred])

            y_pred = model.predict(np.array(x_train))
            results_train.append([key, y_train, y_pred])

            # if i == 0:
            #     vsm_models_visualization.setup_visualization(key, model, feature_names, tech_representation)

    return results_train, results_test
