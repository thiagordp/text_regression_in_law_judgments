"""

"""
import random
import time

import pandas as pd
import tqdm
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile, datapath
from sklearn.model_selection import train_test_split

from evaluation import regression_evaluation
from model import vsm_regression_models, embeddings_regression_models
from representation import bow_tf, bow_tf_idf, word_embeddings
from util.aux_function import print_time
from util.path_constants import MERGE_DATASET, EMBEDDINGS_LIST, EMBEDDINGS_BASE_PATH, INCLUDE_ZERO_VALUES


def test_bow_tf():
    for tests in range(10):
        print("===============================================================================")
        print("Train / Test ", tests)
        raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
        raw_data_df.dropna(inplace=True)

        if not INCLUDE_ZERO_VALUES:
            raw_data_df = raw_data_df.loc[raw_data_df["indenizacao_total"] > 1.0]

        print(raw_data_df["indenizacao_total"].unique())

        # Representation
        x = [row for row in raw_data_df["sentenca"].values]
        y = raw_data_df["indenizacao_total"].values
        # y = y.reshape(-1, 1)

        data = list()
        for i in tqdm.tqdm(range(len(x))):
            row = x[i]
            label = y[i]
            data.append([label, row])

        time.sleep(0.1)
        bow, feature_names = bow_tf.document_vector(x, remove_stopwords=True, stemming=False)

        x_train, x_test, y_train, y_test = train_test_split(bow, y, test_size=0.3, shuffle=True, random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        models_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test,feature_names)
        regression_evaluation.batch_evaluation(models_predictions, independent_vars=len(x_train[0]), description="tf")

    # print("Evaluating")
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, y_poly_pred))
    # r2 = metrics.r2_score(y_test, y_poly_pred)
    # mae = metrics.mean_absolute_error(y_test, y_poly_pred)
    # print(rmse)
    # print(r2)
    # print(mae)


def test_bow_tf_idf():
    for tests in range(10):
        print("===============================================================================")
        print_time()
        print("Train / Test ", tests)
        raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
        raw_data_df.dropna(inplace=True)

        if not INCLUDE_ZERO_VALUES:
            raw_data_df = raw_data_df.loc[raw_data_df["indenizacao_total"] > 1.0]

        print(raw_data_df["indenizacao_total"].unique())

        # Representation
        x = [row for row in raw_data_df["sentenca"].values]
        y = raw_data_df["indenizacao_total"].values
        # y = y.reshape(-1, 1)

        # bow = bow_tf.document_vector(x, remove_stopwords=True, stemming=False)
        bow, feature_names = bow_tf_idf.document_vector(x, remove_stopwords=True, stemming=False)

        x_train, x_test, y_train, y_test = train_test_split(bow, y, test_size=0.3, shuffle=True, random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        models_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test, feature_names)
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
                raw_data_df = raw_data_df.loc[raw_data_df["indenizacao_total"] > 1.0]

            print(raw_data_df["indenizacao_total"].unique())

            # Representation
            x = [row for row in raw_data_df["sentenca"].values]
            y = raw_data_df["indenizacao_total"].astype('float64').values

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
