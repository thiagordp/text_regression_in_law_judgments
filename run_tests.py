"""

"""

import os
import sys
import warnings

import matplotlib

from evaluation.regression_evaluation import feature_relations
from project_tests import test_pre_processing, test_lda, test_paper_experiments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def test_regression():
    # test_representations.test_feature_selection("AVG-EMB")
    # test_representations.test_feature_selection("TF")
    # test_paper_experiments.evaluate_results()

    # test_representations.test_feature_selection("TF-IDF")
    # test_representations.test_feature_selection("Binary")

    #####################################################################
    # test_representations.test_tf_predictions()
    # test_representations.test_bow_tf()
    # test_representations.test_bow_tf_idf()
    # test_representations.test_embeddings_cnn()

    test_paper_experiments.run_experiments("TF")


def test_pre_processings():
    test_pre_processing.test_unify_database()


def test_models():
    return None


def test_evaluations():
    return None


def test_log():
    path_log = "data/overfitting/bigger/"

    # tf_path = path_log + "results_regression_k_100_1000_attr_w_fs_tf_wo_outlier.csv"
    # tf_idf_path = path_log + "results_regression_k_100_1000_tf_idf.csv"
    # binary_path = path_log + "results_regression_k_100_1000_tf_binary.csv"

    # process_overfitting_log(tf_path, "TF")
    # process_overfitting_log(tf_idf_path, "TF-IDF")
    # process_overfitting_log(binary_path, "TF-Binary")
    # regression_evaluation.fix_logs()
    test_paper_experiments.evaluate_results()


def test_paper_results_evaluation():
    test_paper_experiments.paper_results_evaluation()


def test_lda_jec():
    test_lda.jec_lda()


def test_feature_relations():
    feature_relations()


def main():
    warnings.filterwarnings("ignore")

    # test_lda_jec()
    # test_pre_processings()
    # test_regression()
    matplotlib.rcParams['font.family'] = "FreeSerif"
    #test_log()
    #test_feature_relations()


    test_paper_results_evaluation()


if __name__ == "__main__":
    main()
