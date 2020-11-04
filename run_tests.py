"""

"""

import os
import sys
import warnings

from evaluation.regression_logs_evaluation import process_overfitting_log
from project_tests import test_pre_processing, test_lda, test_paper_experiments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def test_representation():
    # test_representations.test_feature_selection("AVG-EMB")
    # test_representations.test_feature_selection("TF")
    test_paper_experiments.run_experiments("TF")

    # test_representations.test_feature_selection("TF-IDF")
    # test_representations.test_feature_selection("Binary")

    #####################################################################
    # test_representations.test_tf_predictions()
    # test_representations.test_bow_tf()
    # test_representations.test_bow_tf_idf()
    # test_representations.test_embeddings_cnn()


def test_pre_processings():
    test_pre_processing.test_unify_database()


def test_models():
    return None


def test_evaluations():
    return None


def test_log():
    path_log = "data/overfitting/bigger/"

    tf_path = path_log + "results_regression_k_100_1000_attr_w_fs_tf_wo_outlier.csv"
    # tf_idf_path = path_log + "results_regression_k_100_1000_tf_idf.csv"
    # binary_path = path_log + "results_regression_k_100_1000_tf_binary.csv"

    process_overfitting_log(tf_path, "TF")
    # process_overfitting_log(tf_idf_path, "TF-IDF")
    # process_overfitting_log(binary_path, "TF-Binary")


def test_lda_jec():
    test_lda.jec_lda()


def main():
    warnings.filterwarnings("ignore")

    # test_lda_jec()
    # test_pre_processings()
    test_representation()
    # test_log()


if __name__ == "__main__":
    main()
