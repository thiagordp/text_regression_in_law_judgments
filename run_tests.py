"""

"""

import os

from evaluation.regression_logs_evaluation import process_overfitting_log
from project_tests import test_representations, test_pre_processing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_representation():
    test_representations.test_tf_feature_selection()
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

    tf_path = path_log + "results_regression_k_100_1000_tf.csv"
    tf_idf_path = path_log + "results_regression_k_100_1000_tf_idf.csv"
    binary_path = path_log + "results_regression_k_100_1000_tf_binary.csv"

    process_overfitting_log(tf_path, "TF")
    process_overfitting_log(tf_idf_path, "TF-IDF")
    process_overfitting_log(binary_path, "TF-Binary")


def main():
    # test_pre_processings()
    # test_representation()
    test_log()


if __name__ == "__main__":
    main()
