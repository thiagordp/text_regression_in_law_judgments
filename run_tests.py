"""

"""
import logging
import os
import sys
import warnings

import matplotlib

from evaluation import regression_paper_evaluation
# from evaluation.regression_evaluation import feature_relations
from evaluation.regression_paper_evaluation import paper_results_evaluation
from project_tests import test_pre_processing, test_lda, test_paper_experiments
from util.path_constants import PATH_LOGS

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
    # feature_relations()
    pass


# TODO: create required folders
def create_required_folders():
    pass


def setup_logging():
    """
    Setup logging to show logs in the screen and save in a file.
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename="data/paper/running_logs/execution_log.log",
                        level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def main():
    # Initial Setup
    setup_logging()
    warnings.filterwarnings("ignore")
    matplotlib.rcParams['font.family'] = "FreeSerif"

    # Data Preparation
    #test_pre_processing.test_unify_database()
    # Run the experiments
    test_paper_experiments.run_experiments("TF")


if __name__ == "__main__":
    main()
