"""

@author Thiago Raulino Dal Pont
@date Oct 30, 2020
"""

# Import Libraries
import gc
import glob
import random
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from sklearn.model_selection import KFold, train_test_split

from evaluation import regression_evaluation
from evaluation.regression_evaluation import get_cross_validation_average, get_binary_code
from model import vsm_regression_models
from model.feature_selections import bow_feature_selection, remove_outliers_iforest
from pre_processing.text_pre_processing import process_judge, process_has_x, process_loss, process_time_delay
from representation import bow_tf, bow_tf_idf, bow_mean_embeddings, bow_binary
from util.path_constants import INCLUDE_ZERO_VALUES
# Run Experiments
from util.value_contants import K_BEST_FEATURE_PAPER


def run_individual_experiments(tech):
    print("#" * 100)
    print("\t\t\t\t\tIndividual RESULTS")

    """
    Exp 1:
        - Include Attributes
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=False,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 2:
        - N-Grams
    """
    # build_test_setup(tech,
    #                  feature_selection=False,
    #                  use_cross_validation=False,
    #                  remove_outliers=False,
    #                  include_attributes=False,
    #                  n_grams=True,
    #                  reduce_models=False,
    #                  fs_after=False)

    """
    Exp 3:
        - Feature Selection
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=False,
                     n_grams=False,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 4:
        - Cross Validation
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=True,
                     remove_outliers=False,
                     include_attributes=False,
                     n_grams=False,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 5:
        - Remove Outliers 
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=True,
                     include_attributes=False,
                     n_grams=False,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 6:
        - Remove Outliers Both
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=False,
                     n_grams=False,
                     overfitting_avoidance=False,
                     remove_outliers_both=True,
                     fs_after=False)


def run_16_experiments_set_fs_ngram(tech):
    print("#" * 100)
    print("\t\t\t\t\t16 EXPERIMENTS RESULTS")

    """
    Experiment 0: 1001000
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=False,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 1: 1001001
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=False,
                     overfitting_avoidance=True,
                     fs_after=False)

    """
    Experiment 2: 1001010
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=True,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 3: 1001011
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     fs_after=False)

    """
    Experiment 4: 1001100
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=False,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 5: 1001101
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=False,
                     overfitting_avoidance=True,
                     fs_after=False)

    """
    Experiment 6: 1001110
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 7: 1001111
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     fs_after=False)

    ##############################################################

    """
    Experiment 8: 1001000
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=False,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 9: 1001001
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=False,
                     overfitting_avoidance=True,
                     fs_after=False)

    """
    Experiment 10: 1001010
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=True,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 11: 1001011
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     fs_after=False)

    """
    Experiment 12: 1001100
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=False,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 13: 1001101
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=False,
                     overfitting_avoidance=True,
                     fs_after=False)

    """
    Experiment 14: 1001110
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=False,
                     fs_after=False)

    """
    Experiment 15: 1001111
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     remove_outliers_both=True,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     fs_after=False)


def run_incremental_experiments(tech):
    print("#" * 100)
    print("\t\t\t\t\tINCREMENTAL RESULTS")
    """
    Exp 1:
        - Include Attributes
        - N-Grams
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=True,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 2:
        - Include Attributes
        - N-Grams
        - Feature Selections
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=True,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 3:
        - Include Attributes
        - N-Grams
        - Feature Selection
        - Reduce Models
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 4:
        - Include Attributes
        - N-Grams
        - Feature Selection
        - Reduce Models
        - Outliers
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=True,
                     fs_after=False)

    """
    Exp 5:
        - Include Attributes
        - N-Grams
        - Feature Selection
        - Reduce Models
        - Cross-Validation
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=True,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)


def combination_experiments(tech):
    print("#" * 100)
    print("\t\t\t\t\tCombination RESULTS")
    """
    Exp 1:
        - N-Grams
        - Feature Selection
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=False,
                     n_grams=True,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 2:
        - Attributes
        - Feature Selection
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=False,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=False,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 3:
        - N-Grams
        - Feature Selection
        - Cross val
        - Include Attr
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=True,
                     remove_outliers=False,
                     include_attributes=True,
                     n_grams=True,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 5:
        - N-Grams
        - Feature Selection
        - Cross Validation
        - Overfitting Avoidance
    """
    build_test_setup(tech,
                     feature_selection=False,
                     use_cross_validation=True,
                     remove_outliers=False,
                     include_attributes=False,
                     n_grams=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 6: 100111
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     n_grams=False,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 7: 101011
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     n_grams=True,
                     include_attributes=False,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 8: 101101
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=False,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 9: 101110
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=False,
                     remove_outliers_both=False,
                     fs_after=False)

    """
    Exp 10: 001111
    """
    build_test_setup(tech,
                     feature_selection=False,
                     remove_outliers=False,
                     n_grams=True,
                     include_attributes=True,
                     use_cross_validation=True,
                     overfitting_avoidance=True,
                     remove_outliers_both=False,
                     fs_after=False)


def run_experiments(tech):
    # run_individual_experiments(tech)

    run_16_experiments_set_fs_ngram(tech)

    # run_incremental_experiments(tech)

    # combination_experiments(tech)

    # print("Waiting...")
    # time.sleep(2 ** 32)
    ################################################
    # Just Feature Selection
    # build_test_setup(tech,
    #                   feature_selection=False,
    #                  use_cross_validation=False,
    #                  remove_outliers=False,
    #                  include_attributes=False,
    #                  n_grams=False)

    ################################################

    # All
    # build_test_setup(tech,
    #                   feature_selection=False,
    #                  use_cross_validation=True,
    #                  remove_outliers=True,
    #                  include_attributes=True,
    #                  n_grams=True,
    #                  overfitting_avoidance=True,
    #                   remove_outliers_both=False,
    #                  fs_after=False,
    #                  make_predictions="ensemble_voting_bg_mlp_gd_xgb")

    # build_test_setup(tech,
    #                   feature_selection=False,
    #                  use_cross_validation=True,
    #                  remove_outliers=False,
    #                  include_attributes=False,
    #                  n_grams=True,
    #                  overfitting_avoidance=False,
    #                  remove_outliers_both=True,
    #                  fs_after=False,
    #                  make_predictions="ensemble_voting_bg_mlp_gd_xgb")

    # build_test_setup(tech,
    #                   feature_selection=False,
    #                  use_cross_validation=True,
    #                  remove_outliers=False,
    #                  include_attributes=False,
    #                  n_grams=True,
    #                  overfitting_avoidance=False,
    #                  remove_outliers_both=True,
    #                  fs_after=False,
    #                  make_predictions="mlp_400_200_100")
    # make_predictions="ensemble_voting_bg_mlp_gd_xgb_wo_or_wo_oa")

    ################################################

    # Data acquisition (Include Attributes)
    # build_test_setup(tech,
    #                 feature_selection=False,
    #                 use_cross_validation=False,
    #                 remove_outliers=False,
    #                 include_attributes=True,
    #                 n_grams=False,
    #                 reduce_models=False)

    ################################################

    # Preprocessing (N-Grams)
    # build_test_setup(tech,
    #                  feature_selection=False,
    #                  use_cross_validation=False,
    #                  remove_outliers=False,
    #                  include_attributes=False,
    #                  n_grams=True,
    #                  reduce_models=False)

    ################################################

    # Representation (Feature Selection, TF)
    # build_test_setup(tech,
    #                  feature_selection=False,
    #                 use_cross_validation=False,
    #                 remove_outliers=False,
    #                 include_attributes=False,
    #                 n_grams=False,
    #                 reduce_models=False)

    ################################################

    # Models (Reduce Models)
    # build_test_setup(tech,
    #                 feature_selection=False,
    #                 use_cross_validation=False,
    #                 remove_outliers=False,
    #                 include_attributes=False,
    #                 n_grams=False,
    #                 reduce_models=True)

    ################################################

    # Models (Cross Validation and Remove Outlier)
    # build_test_setup(tech,
    #                  feature_selection=False,
    #                  use_cross_validation=True,
    #                  remove_outliers=True,
    #                  include_attributes=False,
    #                  n_grams=False,
    #                  reduce_models=False)

    ################################################

    # Training (Attributes, N-grams, Feature Selection)
    # build_test_setup(tech,
    #                   feature_selection=False,
    #                  use_cross_validation=False,
    #                  remove_outliers=False,
    #                  include_attributes=True,
    #                  n_grams=True,
    #                  reduce_models=False)

    ##############################################################################################################################

    # Training (Attributes, N-grams, Feature Selection, Reduce Model)
    # build_test_setup(tech,
    #                   feature_selection=False,
    #                  use_cross_validation=False,
    #                  remove_outliers=False,
    #                  include_attributes=True,
    #                  n_grams=True,
    #                  reduce_models=True)

    ##############################################################################################################################

    print("Waiting")
    time.sleep(2 ** 32 - 1)
    ##############################################################################################################################

    print("Waiting")
    time.sleep(2 ** 32 - 1)


def build_test_setup(tech, feature_selection, use_cross_validation, remove_outliers, include_attributes,
                     n_grams, overfitting_avoidance, remove_outliers_both, fs_after=True, make_predictions=None):
    print("=" * 100)
    print(datetime.today())
    print("PAPER EXPERIMENTS")
    print("Tech:              ", tech)
    print("Feature Selection: ", feature_selection)
    print("Cross Validation:  ", use_cross_validation)
    print("Remove Outliers:   ", remove_outliers)
    print("Include Attributes:", include_attributes)
    print("Include N-Grams:   ", n_grams)
    print("Reduce Models:     ", overfitting_avoidance)
    print("After FS:          ", fs_after)
    print("Remove outlier_both", remove_outliers_both)
    print("")

    # Read CSV with processed documents
    raw_data_df = pd.read_csv("data/processed_dataset_wo_stopwords_wo_stemming.csv")
    print("Processed documents:", raw_data_df.shape[0])

    # Remove NaN values
    raw_data_df.dropna(inplace=True)

    # Remove documentos with Zero values
    if not INCLUDE_ZERO_VALUES:
        raw_data_df = raw_data_df.loc[raw_data_df["indenizacao"] > 1.0]
        print("No proc w/ compensation > 0:", raw_data_df.shape[0])

    # In case you want to filter by judge, uncomment the following lines.
    # raw_data_df["juiz"] = raw_data_df["juiz"].apply(lambda  x: str(x).strip())
    # raw_data_df = raw_data_df.loc[raw_data_df["juiz"] == "Vânia Petermann"]
    # print("After:            ", raw_data_df.shape[0])

    # Convert np.array of texts to list
    x = [row for row in raw_data_df["sentenca"].values]

    # Get Judgments No of each case
    sentenca_std = [str(row) for row in raw_data_df["judgement"].values]

    # Extract attributes
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
    has_no_show_list = process_has_x(raw_data_df["tem_no_show"].values)
    has_overbooking_list = process_has_x(raw_data_df["tem_overbooking"].values)
    has_cancel_refunding_problem_list = process_has_x(raw_data_df["tem_cancelamento_usuario_ressarcimento"].values)
    has_offer_disagreement_list = process_has_x(raw_data_df["tem_desacordo_oferta"].values)

    # Compensation values
    std_y = raw_data_df["indenizacao"].values

    # Get bow representation
    time.sleep(0.1)
    if tech == "TF":
        std_bow, feature_names = bow_tf.document_vector(x, n_grams=n_grams)
    elif tech == "TF-IDF":
        std_bow, feature_names = bow_tf_idf.document_vector(x, n_grams=n_grams)
    elif tech == "AVG-EMB":
        std_bow, feature_names = bow_mean_embeddings.document_vector(x)
    else:  # if tech == "Binary"
        std_bow, feature_names = bow_binary.document_vector(x)

    list_results = list()

    # Convert np.array to list
    list_bow = list(std_bow)

    print("Total documents:", len(list_bow))

    # Clean the memory
    del std_bow
    gc.collect()

    #############################################
    #              feature_selection            #
    #############################################
    if feature_selection and not fs_after:
        list_bow = bow_feature_selection(list_bow, std_y, K_BEST_FEATURE_PAPER)

    if include_attributes:
        for i in range(len(list_bow)):
            # Judgement date infomation
            day = days_list[i]
            month = months_list[i]
            year = years_list[i]
            day_week = day_week_list[i]

            # Judge information
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
            has_overbooking = has_overbooking_list[i]
            has_no_show = has_no_show_list[i]
            has_cancel_refunding = has_cancel_refunding_problem_list[i]
            has_offer_disagreement = has_offer_disagreement_list[i]

            # Include attributes to bag of words
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
            list_bow[i] = np.append(list_bow[i], has_no_show)
            list_bow[i] = np.append(list_bow[i], has_overbooking)
            list_bow[i] = np.append(list_bow[i], has_cancel_refunding)
            list_bow[i] = np.append(list_bow[i], has_offer_disagreement)

            del judge, type_judge, day, month, year, day_week, has_permanent_loss
            del has_temporally_loss, interval_loss, has_luggage_violation
            del has_flight_delay, has_flight_cancellation, flight_delay
            del is_consumers_fault, has_adverse_flight_conditions
            del has_no_show, has_overbooking, has_cancel_refunding, has_offer_disagreement

    del judges, type_judges, days_list, day_week_list, months_list, years_list
    del has_permanent_loss_list, has_temporally_loss_list, interval_loss_list
    del has_luggage_violation_list, has_flight_delay_list, has_flight_cancellation_list
    del flight_delay_list, is_consumers_fault_list, has_adverse_flight_conditions_list
    del has_no_show_list, has_overbooking_list, has_cancel_refunding_problem_list, has_offer_disagreement_list

    gc.collect()

    #############################################
    #              feature_selection            #
    #############################################
    if feature_selection and fs_after:
        bow = bow_feature_selection(list_bow, std_y, K_BEST_FEATURE_PAPER)
    else:
        bow = list_bow

    y = std_y

    sentenca_num = sentenca_std

    repetitions = 10
    if not use_cross_validation:
        repetitions = 100

    if make_predictions is not None:
        repetitions = 1

    if remove_outliers_both:
        bow, y, sentenca_num = remove_outliers_iforest(bow, y, sentenca_num)
        print("No Docs (OR2):", len(y))

    for repetition in tqdm.tqdm(range(repetitions)):
        arr = list()

        for i in range(len(sentenca_num)):
            bow_i = list(bow[i])
            s_i = sentenca_num[i]
            arr.append([s_i, bow_i])

        results_cross_val = list()
        sentence_test_list = list()
        test_predictions_list = list()
        y_test_list = list()

        #############################################
        #              cross_validation             #
        #############################################
        if use_cross_validation:
            random_state = int(str(int((random.random() * random.random() * time.time())))[::-1]) % 2 ** 32
            kfold = KFold(n_splits=3, shuffle=True, random_state=random_state)

            final_set = kfold.split(arr, y)
        else:
            random_state = int((random.random() * random.random() * time.time())) % 2 ** 32

            x_train, x_test, y_train, y_test = train_test_split(arr, y, test_size=0.3, random_state=random_state)

            final_set = [[[x_train, y_train], [x_test, y_test]]]

        for train_ix, test_ix in final_set:

            if use_cross_validation:
                x_train = np.array(arr)[train_ix.astype(int)]
                x_test = np.array(arr)[test_ix.astype(int)]
                y_train = np.array(y)[train_ix.astype(int)]
                y_test = np.array(y)[test_ix.astype(int)]

            else:
                x_train, y_train = train_ix
                x_test, y_test = test_ix

            sentence_test = [int(row[0]) for row in x_test]
            sentence_train = [int(row[0]) for row in x_train]

            x_train = [row[1] for row in x_train]
            x_test = [row[1] for row in x_test]

            #############################################
            #              remove outliers              #
            #############################################
            if remove_outliers:
                x_train, y_train, sentence_train = remove_outliers_iforest(x_train, y_train, sentence_train)

            # Train the models
            train_predictions, test_predictions = vsm_regression_models.full_models_regression(x_train, y_train, x_test, y_test,
                                                                                               feature_names, "tf",
                                                                                               papers_models=True,
                                                                                               reduce_models=overfitting_avoidance)
            # Evaluate the predictions
            dict_results = regression_evaluation.overfitting_evaluation(train_predictions, test_predictions)

            # Get the used k
            if feature_selection:
                k = K_BEST_FEATURE_PAPER
            else:
                k = len(x_train[0])

            for row in dict_results:
                row.append(k)

            results_cross_val.append(dict_results)

            test_predictions_list.extend(list(test_predictions))
            y_test_list.extend(list(y_test))
            sentence_test_list.extend(list(sentence_test))

            # Clean Memory
            del sentence_test, sentence_train, x_train, x_test, y_train, y_test
            del train_predictions, test_predictions

        # Append results to the list
        list_results.extend(get_cross_validation_average(results_cross_val))
        if make_predictions is not None:
            regression_evaluation.save_predictions(tech=tech,
                                                   pred_test=test_predictions_list,
                                                   sentence_test=sentence_test_list,
                                                   output_file_path="data/paper/final/predictions_" + tech + ".@")

        del results_cross_val, test_predictions_list, sentence_test_list

    del sentenca_num

    # Create dataframe using results' list
    df = pd.DataFrame(list_results, columns=["tech", "rmse_test", "rmse_train", "rmse_ratio",
                                             "r2_train", "r2_test", "r2_ratio",
                                             "mae_train", "mae_test", "mae_ratio", "k"])

    # Write file name according to the experimental setup.
    file_name = "data/paper/results_regression_@fs_@tech_@outlier_@n_gram_@attr_@cross_val_@reduce_model_@remove_outliers_both.#"

    file_name = file_name.replace("@tech", str(tech).lower())

    if remove_outliers:
        file_name = file_name.replace("@outlier", "wo_or1")
    else:
        file_name = file_name.replace("@outlier", "w_or1")

    if n_grams:
        file_name = file_name.replace("@n_gram", "w_ng")
    else:
        file_name = file_name.replace("@n_gram", "wo_ng")

    if feature_selection:
        file_name = file_name.replace("@fs", "w_fs_@fs_after_" + str(K_BEST_FEATURE_PAPER))
        if fs_after:
            file_name = file_name.replace("@fs_after", "after")
        else:
            file_name = file_name.replace("@fs_after", "before")
    else:
        file_name = file_name.replace("@fs", "wo_fs")

    if include_attributes:
        file_name = file_name.replace("@attr", "w_at")
    else:
        file_name = file_name.replace("@attr", "wo_at")

    if use_cross_validation:
        file_name = file_name.replace("@cross_val", "w_cv")
    else:
        file_name = file_name.replace("@cross_val", "wo_cv")

    if overfitting_avoidance:
        file_name = file_name.replace("@reduce_model", "w_oa")
    else:
        file_name = file_name.replace("@reduce_model", "wo_oa")

    if remove_outliers_both:
        file_name = file_name.replace("@remove_outliers_both", "w_or2")
    else:
        file_name = file_name.replace("@remove_outliers_both", "wo_or2")

    df.to_csv(file_name.replace("#", "csv"), index=False)
    df.to_excel(file_name.replace("#", "xlsx"), index=False)
    df.to_json(file_name.replace("#", "json"), orient="records")


def evaluate_results():
    """
    Evaluate Results
    """
    skip_techs = [
        "svr_poly_rbf",
        "svr_linear",
        "gradient_boosting",
        "ridge",
        "adaboost",
        "decision_tree",
        "mlp_400_200_100_50",
        "elastic_net",
        "xgboost",
        "xgboost_rf",
        "bagging",
    ]

    logs = glob.glob("data/paper/*.csv")
    fullresults = dict()

    techs = []
    techs_all = []
    ensemble_results = list()
    mlp_results = list()

    for log in logs:
        if log.find("_table") != -1:
            continue

        print("=" * 128)
        print(log)

        df = pd.read_csv(log)
        techs = [tech for tech in df["tech"] if tech not in skip_techs]
        techs = sorted(set(list(techs)))
        techs_all.extend(techs)

        results = list()
        full_results = list()

        for tech in techs:
            if tech in skip_techs:
                continue

            sub_df = df[df["tech"] == tech]

            rmse_test_mean = np.mean(sub_df["rmse_test"])
            rmse_test = np.array(sub_df["rmse_test"])
            mae_test_mean = np.mean(sub_df["mae_test"])
            mae_test = np.array(sub_df["mae_test"])
            r2_test_mean = np.mean(sub_df["r2_test"])
            r2_test = np.array(sub_df["r2_test"])

            if tech.find("emsemble") >= 0 or tech.find("ensemble") >= 0:
                tech = "ensemble_voting"
                bin_code = get_binary_code(log)
                ensemble_results.append([tech, log, bin_code, rmse_test, mae_test, r2_test])

            elif tech[:5].find("mlp") >= 0:
                bin_code = get_binary_code(log)
                mlp_results.append([tech, log, bin_code, rmse_test, mae_test, r2_test])

            results.append([tech, rmse_test_mean, mae_test_mean, r2_test_mean])
            full_results.append([tech, log, rmse_test, mae_test, r2_test])

        df = pd.DataFrame(results, columns=["tech", "rmse_test", "mae_test", "r2_test"])
        df_full = pd.DataFrame(full_results, columns=["tech", "log", "rmse_test", "mae_test", "r2_test"])

        fullresults[log] = df

        plot_metrics(df, log)
        # plot_violin(df_full, log, x_col="tech")

    df_ensemble = pd.DataFrame(ensemble_results, columns=["tech", "log", "bin_code", "rmse_test", "mae_test", "r2_test"])
    df_mlp = pd.DataFrame(mlp_results, columns=["tech", "log", "bin_code", "rmse_test", "mae_test", "r2_test"])
    plot_violin_box_plot(df_ensemble, log="", x_col="bin_code", file_name="data/paper/ensemble_combinations_boxplot.#", plot_violin=False)
    plot_violin_box_plot(df_mlp, log="", x_col="bin_code", file_name="data/paper/mlp_combinations_boxplot.#", plot_violin=False)

    regression_evaluation.build_binary_table(logs, sorted(set(techs_all)))


def plot_violin_box_plot(df_full, log="", x_col="", file_name="", plot_violin=True):
    final_data = list()

    for index, row in df_full.iterrows():
        log_path = row["log"]
        bin_code = row["bin_code"]
        tech = row["tech"]
        rmse_test = list(row["rmse_test"])
        mae_test = list(row["mae_test"])
        r2_test = list(row["r2_test"])

        # Shuffle is not a must
        random.shuffle(rmse_test)
        random.shuffle(mae_test)
        random.shuffle(r2_test)

        for i in range(len(rmse_test)):
            final_data.append([tech, bin_code, rmse_test[i], r2_test[i], mae_test[i]])

        # time.sleep(5)

    new_df = pd.DataFrame(final_data, columns=["tech", "bin_code", "rmse_test", "r2_test", "mae_test"])

    plt.close()
    plt.figure(figsize=(15, 8))
    plt.grid(linestyle=':', linewidth=2, axis="y")

    if plot_violin:

        sns.violinplot(y="r2_test",
                       x=x_col,
                       data=new_df)
    else:
        sns.boxplot(y="r2_test",
                    x=x_col,
                    data=new_df)

    plt.title("R2 Test " + tech)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    if file_name is None:
        fn = log.replace(".csv", "_r2_violin.png")
    else:
        fn = file_name.replace(".#", "_r2.png")

    plt.savefig(fn, dpi=300)

    plt.close()
    plt.figure(figsize=(15, 8))

    plt.title("RMSE Test " + tech)
    plt.xticks(rotation='vertical')
    plt.grid(linestyle=':', linewidth=2, axis="y")

    if plot_violin:
        sns.violinplot(y="rmse_test",
                       x=x_col,
                       data=new_df)
    else:
        sns.boxplot(y="rmse_test",
                    x=x_col,
                    data=new_df)

    if file_name is None:
        fn = log.replace(".csv", "_rmse_violin.png")
    else:
        fn = file_name.replace(".#", "_rmse.png")

    plt.tight_layout()
    plt.savefig(fn, dpi=300)

    plt.close()
    plt.figure(figsize=(15, 8))

    plt.title("MAE Test " + tech)
    plt.xticks(rotation='vertical')
    plt.grid(linestyle=':', linewidth=2, axis="y")
    if plot_violin:
        sns.violinplot(y="mae_test",
                       x=x_col,
                       data=new_df)
    else:
        sns.boxplot(y="mae_test",
                    x=x_col,
                    data=new_df)

    plt.tight_layout()

    if file_name is None:
        fn = log.replace(".csv", "_mae_violin.png")
    else:
        fn = file_name.replace(".#", "_mae.png")

    plt.savefig(fn, dpi=300)


def plot_violin_2(df_full, log):
    final_data = list()

    for index, row in df_full.iterrows():
        log_path = row["log"]
        tech = row["tech"]
        rmse_test = list(row["rmse_test"])
        mae_test = list(row["mae_test"])
        r2_test = list(row["r2_test"])
        #
        # print(len(rmse_test))
        # print(len(r2_test))
        # print(len(mae_test))

        for i in range(len(rmse_test)):
            final_data.append([tech, log, rmse_test[i], r2_test[i], mae_test[i]])

        # time.sleep(5)

    new_df = pd.DataFrame(final_data, columns=["tech", "log", "rmse_test", "r2_test", "mae_test"])

    plt.title("RMSE Test")
    plt.xticks(rotation='vertical')
    plt.grid(linestyle=':', linewidth=2, axis="y")
    plt.tight_layout()
    plt.savefig(log.replace(".csv", "_rmse_violin.png"))


def plot_metrics(results, log):
    techs = sorted(set(results["tech"]))
    matplotlib.style.use("seaborn")

    plt.figure(figsize=(15, 10))
    plt.grid(linestyle=':')

    r2_mean = list()

    table_results_list = list()

    for tech in techs:
        data = results[results["tech"] == tech]["r2_test"]
        r2_mean.append(float(data))
        plt.bar(tech, data)

    data = results["r2_test"]
    print("Mean R2:", round(float(np.mean(data)), 4))

    for tech, data_tech in zip(techs, data):
        label = "{:.4f}".format(data_tech)

        plt.annotate(label,  # this is the text
                     (tech, data_tech),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     fontsize=14,
                     ha='center')  # horizontal alignment can be left, right or center

    plt.xticks(rotation='vertical')
    plt.title("R2 Test", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(bbox_to_anchor=(0.5, -0.1), ncol=4, loc='upper center', borderaxespad=0., fontsize=14)

    plt.tight_layout()
    plt.savefig(log.replace(".csv", "_r2_test.png"))

    #########################################################

    plt.figure(figsize=(15, 10))
    plt.grid(linestyle=':')

    # RMSE plot
    for tech in techs:
        data = results[results["tech"] == tech]["rmse_test"]
        plt.bar(tech, data)

    data = results["rmse_test"]
    print("Mean RMSE:", round(float(np.mean(data)), 2))
    for tech, data_tech in zip(techs, data):
        label = "{:.2f}".format(data_tech)

        plt.annotate(label,  # this is the text
                     (tech, data_tech),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     fontsize=14,
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.xticks(rotation='vertical')
    plt.title("RMSE Test", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(bbox_to_anchor=(0.5, -0.1), ncol=4, loc='upper center', borderaxespad=0., fontsize=14)

    plt.tight_layout()
    plt.savefig(log.replace(".csv", "_rmse_test.png"))

    plt.figure(figsize=(15, 10))
    plt.grid(linestyle=':')

    # MAE plot
    for tech in techs:
        data = results[results["tech"] == tech]["mae_test"]
        plt.bar(tech, data)

    data = results["mae_test"]
    print("Mean MAE:", round(float(np.mean(data)), 2))
    for tech, data_tech in zip(techs, data):
        label = "{:.2f}".format(data_tech)

        plt.annotate(label,  # this is the text
                     (tech, data_tech),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     fontsize=14,
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.xticks(rotation='vertical')
    plt.title("MAE Test", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(bbox_to_anchor=(0.5, -0.1), ncol=4, loc='upper center', borderaxespad=0., fontsize=14)

    plt.tight_layout()
    plt.savefig(log.replace(".csv", "_mae_test.png"))

    metrics = ["r2_test", "rmse_test", "mae_test"]

    data = list()
    techs = sorted(set(results["tech"]))
    columns = ["metric"]
    columns = columns + techs

    for metric in metrics:
        row_data = list()
        row_data.append(metric)

        for tech in techs:
            metric_value = results[results["tech"] == tech][metric]

            row_data.append(float(metric_value))

        data.append(row_data)

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(log.replace(".csv", "_table.csv"), index=False)
    df.to_excel(log.replace(".csv", "_table.xlsx"), index=False)