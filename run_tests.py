"""

"""

import os
import sys
import warnings

import matplotlib

from evaluation import regression_paper_evaluation
from evaluation.regression_evaluation import feature_relations
from project_tests import test_pre_processing, test_lda, test_paper_experiments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def test_regression():

    test_paper_experiments.run_experiments("TF")
    test_paper_experiments.evaluate_results()


def test_pre_processings():
    test_pre_processing.test_unify_database()


def test_feature_relations():
    feature_relations()


def main():
    warnings.filterwarnings("ignore")

    matplotlib.rcParams['font.family'] = "FreeSerif"

    test_pre_processing.test_unify_database()
    test_paper_experiments.run_experiments("TF")
    test_paper_experiments.evaluate_results()
    regression_paper_evaluation.paper_results_evaluation()


if __name__ == "__main__":
    main()
