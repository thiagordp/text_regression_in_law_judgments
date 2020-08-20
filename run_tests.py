"""

"""

import os

from project_tests import test_representations, test_pre_processing
from project_tests.test_log_evaluation import test_log_evaluation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_representation():
    test_representations.test_bow_tf()
    # test_representations.test_bow_tf_idf()
    # test_representations.test_embeddings_cnn()


def test_pre_processings():
    test_pre_processing.test_unify_database()


def test_models():
    return None


def test_evaluations():
    return None


def test_log():
    test_log_evaluation()


def main():
    # test_pre_processings()
    test_representation()
    # test_log()


if __name__ == "__main__":
    main()
