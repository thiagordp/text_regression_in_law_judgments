"""
@description
    Visualization of Vector Space Models for Text Regression
@author
    Thiago Raulino Dal Pont
"""

import graphviz
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

from util.path_constants import PROJECT_PATH


def setup_visualization(model_name, model, feature_names, tech_representation):
    if model_name.find("decision_tree") != -1:
        visualize_decision_tree(model, feature_names, tech_representation)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        columns = ["feature_name", "importance"]
        list_importances = list()
        for i in range(len(feature_names)):
            list_importances.append([feature_names[indices[i]], importances[indices[i]]])

        df = pd.DataFrame(data=list_importances, columns=columns)
        file_name = "data/features/decision_tree_@.csv".replace("@", tech_representation)
        df.to_csv(file_name)
        df.to_excel(file_name.replace(".csv", ".xlsx"))

    elif model_name.find("random_forest") != -1:

        visualize_random_forest(model, feature_names, tech_representation)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        columns = ["feature_name", "importance"]
        list_importances = list()
        for i in range(len(feature_names)):
            list_importances.append([feature_names[indices[i]], importances[indices[i]]])

        df = pd.DataFrame(data=list_importances, columns=columns)
        file_name = "data/features/random_forest_@.csv".replace("@", tech_representation)

        df.to_csv(file_name)
        df.to_excel(file_name.replace(".csv", ".xlsx"))


def visualize_decision_tree(model=DecisionTreeRegressor(), features_names=list(), type_representation=""):
    """
    Decision Tree visualization
    :param model: Trained Model
    :param features_names: Names of the features
    :return: None
    """
    file_name = PROJECT_PATH + "data/figures/decision_treee_$.dot"
    file_name = file_name.replace("$", type_representation)

    # export_graphviz(model, feature_names=features_names, filled=True, rounded=True)
    dot_data = export_graphviz(model, out_file=None, feature_names=features_names, rounded=True, proportion=False, precision=2, filled=True)
    graph = graphviz.Source(dot_data)
    graph.render(file_name.replace(".dot", ""), view=False, format="png")

    # time.sleep(120)


def visualize_random_forest(model=RandomForestRegressor(), feature_names=list(), type_representation=""):
    """
    Random Forest visualization
    :param model: Trained Model
    :param feature_names: List of Features
    :param type_representation: Type of Representation (TF or TF-IDF)
    """

    file_name = PROJECT_PATH + "data/figures/random_forest_$_@.dot"
    file_name = file_name.replace("$", type_representation)

    for i, tree in enumerate(model.estimators_):
        f_name = file_name.replace("@", str(i).zfill(2))
        dot_data = export_graphviz(tree, out_file=None, feature_names=feature_names, rounded=True, proportion=False, precision=2, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render(f_name.replace(".dot", ""), view=False, format="png")


def visualize_cnn(model):
    return None


def visualize_biggest_erros(model, y_test, y_pred):
    return None

