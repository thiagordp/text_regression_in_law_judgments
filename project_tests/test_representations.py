"""

"""
import random
import time

import numpy as np
import pandas as pd
import tqdm
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile, datapath
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from evaluation import regression_evaluation
from evaluation.regression_evaluation import get_cross_validation_average
from model import vsm_regression_models, embeddings_regression_models
from model.feature_selections import bow_feature_selection
from representation import bow_tf, bow_tf_idf, word_embeddings, bow_binary
from util.aux_function import print_time
from util.path_constants import MERGE_DATASET, EMBEDDINGS_LIST, EMBEDDINGS_BASE_PATH, INCLUDE_ZERO_VALUES, PROCESSED_DATASET_W_SW
from util.value_contants import K_BEST_FEATURES_LIST


def test_tf_feature_selection():
    print("===============================================================================")
    print("Train / Test ")
    raw_data_df = pd.read_csv(PROCESSED_DATASET_W_SW, index_col=0)
    raw_data_df.dropna(inplace=True)

    if not INCLUDE_ZERO_VALUES:
        raw_data_df = raw_data_df.loc[raw_data_df["indenizacao"] > 1.0]

    # Representation
    x = [row for row in raw_data_df["sentenca"].values]
    sentenca_num = [str(row) for row in raw_data_df["judgement"].values]

    print(raw_data_df["indenizacao"].unique())
    y = raw_data_df["indenizacao"].values

    data = list()
    for i in tqdm.tqdm(range(len(x))):
        row = x[i]
        label = y[i]
        data.append([label, row])

    time.sleep(0.1)
    std_bow, feature_names = bow_binary.document_vector(x, remove_stopwords=True, stemming=False)
    list_results = list()

    for k in K_BEST_FEATURES_LIST:

        print("--------------------------------------------")
        print_time()
        print("K", k)

        bow = bow_feature_selection(std_bow, y, k)
        bow = list(bow)

        for repetition in tqdm.tqdm(range(5)):
            arr = list()

            for i in range(len(sentenca_num)):
                bow_i = list(bow[i])
                s_i = sentenca_num[i]
                arr.append([s_i, bow_i])

            kfold = KFold(n_splits=10,
                          shuffle=True,
                          random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)
            results_cross_val = list()
            for train_ix, test_ix in kfold.split(arr, y):
                x_train = np.array(arr)[train_ix.astype(int)]
                x_test = np.array(arr)[test_ix.astype(int)]
                y_train = np.array(y)[train_ix.astype(int)]
                y_test = np.array(y)[test_ix.astype(int)]



                x_train = [row[1] for row in x_train]
                x_test = [row[1] for row in x_test]

                train_predictions, test_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test, feature_names, "tf")
                dict_results = regression_evaluation.overfitting_evaluation(train_predictions, test_predictions)

                for row in dict_results:
                    row.append(k)

                results_cross_val.append(dict_results)


            list_results.extend(get_cross_validation_average(results_cross_val))

        # regression_evaluation.batch_evaluation(train_predictions, test_predictions, sentence_train, sentence_test, description="tf")
        # tech, rmse_train, rmse_test, rmse_ratio, r2_train, r2_test, r2_ratio, mae_train, mae_test, mae_ratio
        df = pd.DataFrame(list_results, columns=["tech", "rmse_test", "rmse_train", "rmse_ratio",
                                                 "r2_train", "r2_test", "r2_ratio",
                                                 "mae_train", "mae_test", "mae_ratio", "k"])
        df.to_csv("data/overfitting/results_regression_k_100_1000.csv")
        df.to_excel("data/overfitting/results_regression_k_100_1000.xlsx")


def test_bow_tf():
    for tests in range(1):
        print("===============================================================================")
        print("Train / Test ", tests)
        raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
        raw_data_df.dropna(inplace=True)

        if not INCLUDE_ZERO_VALUES:
            raw_data_df = raw_data_df.loc[raw_data_df["indenizacao"] > 1.0]

        # Representation
        x = [row for row in raw_data_df["sentenca"].values]
        sentenca_num = [str(row) for row in raw_data_df["judgement"].values]

        print(raw_data_df["indenizacao"].unique())
        y = raw_data_df["indenizacao"].values

        data = list()
        for i in tqdm.tqdm(range(len(x))):
            row = x[i]
            label = y[i]
            data.append([label, row])

        time.sleep(0.1)
        bow, feature_names = bow_tf.document_vector(x, remove_stopwords=True, stemming=False)

        print(bow.shape)
        arr = list()
        bow = list(bow)

        for i in range(len(sentenca_num)):
            bow_i = list(bow[i])
            s_i = sentenca_num[i]
            arr.append([s_i, bow_i])

        kfold = KFold(n_splits=10,
                      shuffle=True,
                      random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        x_train, x_test, y_train, y_test = train_test_split(arr, y, test_size=0.3, shuffle=True, random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        sentence_train = [row[0] for row in x_train]
        x_train = [row[1] for row in x_train]

        sentence_test = [row[0] for row in x_test]
        x_test = [row[1] for row in x_test]

        train_predictions, test_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test, feature_names, "tf")
        regression_evaluation.batch_evaluation(train_predictions, test_predictions, sentence_train, sentence_test, description="tf")

    # print("Evaluating")
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, y_poly_pred))
    # r2 = metrics.r2_score(y_test, y_poly_pred)
    # mae = metrics.mean_absolute_error(y_test, y_poly_pred)
    # print(rmse)
    # print(r2)
    # print(mae)


def test_bow_tf_idf():
    for tests in range(1):
        print("===============================================================================")
        print_time()
        print("Train / Test ", tests)
        raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
        raw_data_df.dropna(inplace=True)

        if not INCLUDE_ZERO_VALUES:
            raw_data_df = raw_data_df.loc[raw_data_df["indenizacao"] > 1.0]

        print(raw_data_df["indenizacao"].unique())

        # Representation
        x = [row for row in raw_data_df["sentenca"].values]
        y = raw_data_df["indenizacao"].values
        # y = y.reshape(-1, 1)

        # bow = bow_tf.document_vector(x, remove_stopwords=True, stemming=False)
        bow, feature_names = bow_tf_idf.document_vector(x, remove_stopwords=True, stemming=False)

        x_train, x_test, y_train, y_test = train_test_split(bow, y, test_size=0.3, shuffle=True, random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        models_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test, feature_names, "tf_idf")
        regression_evaluation.batch_evaluation(models_predictions, independent_vars=len(x_train[0]), description="tf_idf")


def test_embeddings_cnn():
    for emb_path in EMBEDDINGS_LIST:
        print("---------------------- EMBEDDINGS", emb_path, "----------------------")
        emb = EMBEDDINGS_BASE_PATH + emb_path

        # Load embeddings
        if emb.find("glove") != -1:
            glove_file = datapath(emb)
            emb = get_tmpfile("glove2word2vec.txt")
            glove2word2vec(glove_file, emb)

        for tests in range(10):
            print("Train / Test ", tests)
            print_time()
            raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
            raw_data_df.dropna(inplace=True)

            if not INCLUDE_ZERO_VALUES:
                raw_data_df = raw_data_df.loc[raw_data_df["indenizacao"] > 1.0]

            print(raw_data_df["indenizacao"].unique())

            # Representation
            x = [row for row in raw_data_df["sentenca"].values]
            y = raw_data_df["indenizacao"].astype('float64').values

            word_vectors = KeyedVectors.load_word2vec_format(emb, binary=False)

            # Get matrix representations
            x, vocabulary_size, emb_len, embeddings_matrix = word_embeddings.document_matrix(x, word_vectors=word_vectors)

            # Split train, test, validation sets
            random_state = int((random.random() * random.random() * random.random() * time.time())) % 2 ** 32
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=random_state)
            random_state = int((random.random() * random.random() * random.random() * time.time())) % 2 ** 32
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=random_state)

            models_predictions = embeddings_regression_models.simple_cnn_model(x_train, y_train, x_test, y_test, x_val, y_val, vocabulary_size, emb_len, embeddings_matrix,
                                                                               emb_path)
            regression_evaluation.batch_evaluation(models_predictions, independent_vars=100, description="cnn")


def test_tf_predictions():
    x = 0