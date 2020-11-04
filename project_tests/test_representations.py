"""

"""
import gc
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
from evaluation.regression_evaluation import get_cross_validation_average, overfitting_prediction
from model import vsm_regression_models, embeddings_regression_models
from model.feature_selections import bow_feature_selection, remove_outliers
from pre_processing.text_pre_processing import process_judge, process_has_x, process_loss, process_time_delay
from representation import bow_tf, bow_tf_idf, word_embeddings, bow_binary, bow_mean_embeddings
from util.aux_function import print_time
from util.path_constants import MERGE_DATASET, EMBEDDINGS_LIST, EMBEDDINGS_BASE_PATH, INCLUDE_ZERO_VALUES
from util.value_contants import K_BEST_FEATURES_LIST, SAVE_PREDICTIONS, INCLUDE_ATTRIBUTES, REMOVE_OUTLIERS, FEATURE_SELECTION


def test_feature_selection(tech):
    print("===============================================================================")
    print("Train / Test ")

    if tech == "AVG-EMB":
        raw_data_df = pd.read_csv("data/processed_dataset_w_stopwords_wo_stemming.csv")
    else:
        raw_data_df = pd.read_csv("data/processed_dataset_wo_stopwords_wo_stemming.csv")
    print("Processed documents:", raw_data_df.shape[0])

    raw_data_df.dropna(inplace=True)

    if not INCLUDE_ZERO_VALUES:
        raw_data_df = raw_data_df.loc[raw_data_df["indenizacao"] > 1.0]

    # Representation
    x = [row for row in raw_data_df["sentenca"].values]
    # dates = [row for row in raw_data_df["data"].values]
    # judges = [row for row in raw_data_df["judges"].values]
    sentenca_std = [str(row) for row in raw_data_df["judgement"].values]

    days_list = list(raw_data_df["dia"])
    months_list = list(raw_data_df["mes"])
    years_list = list(raw_data_df["ano"])
    day_week_list = list(raw_data_df["dia_semana"])
    judges = list(raw_data_df["juiz"])
    type_judges = list(raw_data_df["tipo_juiz"])

    judges, type_judges = process_judge(judges, type_judges)

    has_permanent_loss_list = process_has_x(raw_data_df["extravio_permanente"].values)
    has_temporally_loss_list = process_has_x(raw_data_df["extravio_temporario"].values)
    interval_loss_list = process_loss(raw_data_df["intevalo_extravio"].values)
    has_luggage_violation_list = process_has_x(raw_data_df["tem_violacao_bagagem"].values)
    has_flight_delay_list = process_has_x(raw_data_df["tem_atraso_voo"].values)
    has_flight_cancellation_list = process_has_x(raw_data_df["tem_cancelamento_voo"].values)
    flight_delay_list = process_time_delay(raw_data_df["qtd_atraso_voo"].values)
    is_consumers_fault_list = process_has_x(raw_data_df["culpa_consumidor"].values)
    has_adverse_flight_conditions_list = process_has_x(raw_data_df["tem_condicao_adversa_voo"].values)

    std_y = raw_data_df["indenizacao"].values

    time.sleep(0.1)
    if tech == "TF":
        std_bow, feature_names = bow_tf.document_vector(x)
    elif tech == "TF-IDF":
        std_bow, feature_names = bow_tf_idf.document_vector(x)
    elif tech == "AVG-EMB":
        std_bow, feature_names = bow_mean_embeddings.document_vector(x)
    else:  # if tech == "Binary"
        std_bow, feature_names = bow_binary.document_vector(x)

    list_results = list()

    list_bow = list(std_bow)
    print("Total documents:", len(list_bow))

    del std_bow
    gc.collect()

    # std_bow = [list(row) for row in std_bow]

    # Concatenate new features to bag of words
    # Include features before selection
    if INCLUDE_ATTRIBUTES:
        for i in range(len(list_bow)):
            day = days_list[i]
            month = months_list[i]
            year = years_list[i]
            day_week = day_week_list[i]
            judge = judges[i]
            type_judge = type_judges[i]

            has_permanent_loss = has_permanent_loss_list[i]
            has_temporally_loss = has_temporally_loss_list[i]
            interval_loss = interval_loss_list[i]
            has_luggage_violation = has_luggage_violation_list[i]
            has_flight_delay = has_flight_delay_list[i]
            has_flight_cancellation = has_flight_cancellation_list[i]
            flight_delay = flight_delay_list[i]
            is_consumers_fault = is_consumers_fault_list[i]
            has_adverse_flight_conditions = has_adverse_flight_conditions_list[i]

            list_bow[i] = np.append(list_bow[i], [day, month, year, day_week])
            list_bow[i] = np.append(list_bow[i], judge)
            list_bow[i] = np.append(list_bow[i], type_judge)
            list_bow[i] = np.append(list_bow[i], has_permanent_loss)
            list_bow[i] = np.append(list_bow[i], has_temporally_loss)
            list_bow[i] = np.append(list_bow[i], [interval_loss, flight_delay])
            list_bow[i] = np.append(list_bow[i], has_luggage_violation)
            list_bow[i] = np.append(list_bow[i], has_flight_delay)
            list_bow[i] = np.append(list_bow[i], has_flight_cancellation)
            list_bow[i] = np.append(list_bow[i], is_consumers_fault)
            list_bow[i] = np.append(list_bow[i], has_adverse_flight_conditions)

            del judge, type_judge, day, month, year, day_week, has_permanent_loss
            del has_temporally_loss, interval_loss, has_luggage_violation
            del has_flight_delay, has_flight_cancellation, flight_delay
            del is_consumers_fault, has_adverse_flight_conditions

    del judges, type_judges, days_list, day_week_list, months_list, years_list
    gc.collect()

    for k in K_BEST_FEATURES_LIST:

        print("-" * 150)
        print_time()
        print("K", k)

        if FEATURE_SELECTION:
            bow = bow_feature_selection(list_bow, std_y, k)
            del list_bow
        else:
            bow = list_bow

        y = std_y

        sentenca_num = sentenca_std

        sum_lens = 0

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

            sentence_test_list = list()
            test_predictions_list = list()

            for train_ix, test_ix in kfold.split(arr, y):
                x_train = np.array(arr)[train_ix.astype(int)]
                x_test = np.array(arr)[test_ix.astype(int)]
                y_train = np.array(y)[train_ix.astype(int)]
                y_test = np.array(y)[test_ix.astype(int)]

                sentence_test = [int(row[0]) for row in x_test]
                sentence_train = [row[0] for row in x_train]

                sentence_test_list.extend(sentence_test)

                x_train = [row[1] for row in x_train]
                x_test = [row[1] for row in x_test]

                l1 = len(x_train)
                if REMOVE_OUTLIERS:
                    x_train, y_train, sentence_train = remove_outliers(x_train, y_train, sentence_train)
                l2 = len(x_train)
                sum_lens += l2

                train_predictions, test_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test, feature_names, "tf")
                dict_results = regression_evaluation.overfitting_evaluation(train_predictions, test_predictions)

                for row in dict_results:
                    row.append(k)

                results_cross_val.append(dict_results)
                test_predictions_list.extend(test_predictions)

                del sentence_test, sentence_train, x_train, x_test, y_train, y_test
                del train_predictions, test_predictions

            if SAVE_PREDICTIONS:
                overfitting_prediction(sentence_test_list, test_predictions_list)
                del sentence_test_list, test_predictions_list

            list_results.extend(get_cross_validation_average(results_cross_val))

            del results_cross_val

        print("Average w/o outliers:", round(sum_lens / 50, 2))
        del sentenca_num,

        # regression_evaluation.batch_evaluation(train_predictions, test_predictions, sentence_train, sentence_test, description="tf")
        # tech, rmse_train, rmse_test, rmse_ratio, r2_train, r2_test, r2_ratio, mae_train, mae_test, mae_ratio
        df = pd.DataFrame(list_results, columns=["tech", "rmse_test", "rmse_train", "rmse_ratio",
                                                 "r2_train", "r2_test", "r2_ratio",
                                                 "mae_train", "mae_test", "mae_ratio", "k"])

        file_name = "data/overfitting/bigger/results_regression_k_100_1000_&_@_$.#"
        file_name = file_name.replace("@", str(tech).lower())
        # file_name = file_name.replace("&", "attr_wo_fs")

        if REMOVE_OUTLIERS:
            file_name = file_name.replace("$", "wo_outlier")
        else:
            file_name = file_name.replace("$", "w_outlier")

        if FEATURE_SELECTION:
            file_name = file_name.replace("&", "attr_w_fs")
        else:
            file_name = file_name.replace("&", "attr_wo_fs")

        df.to_csv(file_name.replace("#", "csv"))
        df.to_excel(file_name.replace("#", "xlsx"))
        df.to_json(file_name.replace("#", "json"), orient="records")


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
