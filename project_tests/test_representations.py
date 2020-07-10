"""

"""
import random
import time

import tqdm
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from evaluation import regression_evaluation
from model import vsm_regression_models
from representation import bow_tf, bow_tf_idf
from nltk.corpus import stopwords

from util.path_constants import MERGE_DATASET


def test_bow_tf():
    for tests in range(10):
        print("===============================================================================")
        print("Train / Test ", tests)
        raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
        raw_data_df.dropna(inplace=True)
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
        bow = bow_tf.document_vector(x, remove_stopwords=True, stemming=False)

        x_train, x_test, y_train, y_test = train_test_split(bow, y, test_size=0.3, shuffle=True, random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        models_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test)
        regression_evaluation.batch_evaluation(models_predictions, independent_vars=len(x_train[0]))

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
        print("Train / Test ", tests)
        raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)
        raw_data_df.dropna(inplace=True)
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
        # bow = bow_tf.document_vector(x, remove_stopwords=True, stemming=False)
        bow = bow_tf_idf.document_vector(x, remove_stopwords=True, stemming=False)

        x_train, x_test, y_train, y_test = train_test_split(bow, y, test_size=0.3, shuffle=True, random_state=int((random.random() * random.random() * time.time())) % 2 ** 32)

        models_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test)
        regression_evaluation.batch_evaluation(models_predictions, independent_vars=len(x_train[0]))
