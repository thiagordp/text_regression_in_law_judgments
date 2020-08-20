"""
@description
    Visualization of Vector Space Models for Text Regression
@author
    Thiago Raulino Dal Pont
"""
import random
import time

import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

from util.path_constants import PROJECT_PATH


def visualize_decision_tree(model=DecisionTreeRegressor(), features_names=list()):
    """
    Decision Tree visualization
    :param model: Trained Model
    :param features_names: Names of the features
    :return: None
    """
    file_num = int(random.random() * 10000)
    file_name = PROJECT_PATH + "data/figures/tree_" + str(file_num).zfill(5) + ".dot"

    # export_graphviz(model, feature_names=features_names, filled=True, rounded=True)
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=features_names,
                               rounded=True, proportion=False,
                               precision=2, filled=True)
    graph = graphviz.Source(dot_data)
    graph.render(file_name.replace(".dot", ""), view=False, format="png")

    # time.sleep(120)


def visualize_random_forest(model=RandomForestRegressor()):
    return None


def visualize_cnn(model):
    return None


def visualize_biggest_erros(model, y_test, y_pred):
    return None
