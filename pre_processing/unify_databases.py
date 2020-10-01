"""

"""
import glob
import math
from datetime import datetime

import pandas as pd
import tqdm

from pre_processing.text_pre_processing import process_text
from util.path_constants import *
from util.value_contants import CLASSE_DICT, GET_INDIVIDUAL_VALUES


def create_final_dataset():
    final_data = list()
    attributes_list = list()
    jec_list = list()

    print("Loading JEC judgements")
    for jec_class in tqdm.tqdm(JEC_CLASS_PATHS):
        class_path = JEC_DATASET_PATH + jec_class
        judgements_file_list = glob.glob(class_path + "*.txt")

        for judgement_file in judgements_file_list:
            text = open(judgement_file).read().replace("\n", " ").replace("  ", " ")

            file_name = judgement_file.replace(JEC_DATASET_PATH, "").replace(jec_class, "").replace(".txt", "").replace("/", "")
            jec_class = jec_class.replace("/", "")
            jec_list.append([file_name, jec_class, text])
            # print(file_name, jec_class, text)

    print("Loadings attributes file")
    if GET_INDIVIDUAL_VALUES:
        attributes_df = pd.read_csv(DAMAGE_VALUES_DATASET_PATH,
                                    usecols=["Sentença", "Julgamento", "Valor individual do dano moral", "Data do Julgamento", "Julgador(a)", "Tipo Julgador(a)"])
        attributes_df.dropna(subset=["Julgamento"], inplace=True)
        attributes_df.sort_values('Valor individual do dano moral')

        attributes_df["Julgamento"] = attributes_df["Julgamento"].apply(lambda x: CLASSE_DICT[x])

    else:
        attributes_df = pd.read_csv(DAMAGE_VALUES_DATASET_PATH, usecols=["Sentença", "Julgamento", "Valor total do dano moral"])
        attributes_df.dropna(subset=["Julgamento"], inplace=True)
        attributes_df.sort_values('Valor total do dano moral').drop_duplicates(subset=['Sentença', 'Julgamento'], keep='last', inplace=True)

        attributes_df["Julgamento"] = attributes_df["Julgamento"].apply(lambda x: CLASSE_DICT[x])

    last_class = ""
    last_num = "0"

    attributes_df = attributes_df[attributes_df["Data do Julgamento"].notna()]
    attributes_df = attributes_df[attributes_df["Data do Julgamento"].notnull()]

    # df = df[df['EPS'].notna()]

    print("Merging text and attributes")
    for index, row in tqdm.tqdm(attributes_df.iterrows()):

        num_judgement = row["Sentença"]
        jec_class = row["Julgamento"]
        date = row["Data do Julgamento"]

        # if math.isnan(date):
        #     print("Skipping", num_judgement, "NaN detected")
        #     continue

        format_string = "%d/%m/%Y"
        date = datetime.strptime(date, format_string)
        year = date.year
        month = date.month
        day = date.day
        weekday = date.weekday()
        judge = row["Julgador(a)"]
        type_judge = row["Tipo Julgador(a)"]

        if GET_INDIVIDUAL_VALUES:
            indenizacao = float(str(row["Valor individual do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))
        else:
            indenizacao = float(str(row["Valor total do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))

        attributes_list.append([num_judgement, jec_class, indenizacao])

        for jec_judge in jec_list:
            # print(num_judgement, jec_class, indenizacao, jec_judge)
            int_n = int(num_judgement)
            if int(jec_judge[0]) == int_n:  # and int_n != last_num:
                # jec_judge[1] == jec_class and \
                # (int(last_num) != int(num_judgement) or last_class != jec_class):
                processed_text = process_text(jec_judge[2], remove_stopwords=True, stemming=False)
                final_data.append([int(num_judgement), jec_class, year, month, day, weekday, judge, type_judge, indenizacao, processed_text])

                break

        last_class = jec_class
        last_num = num_judgement

    final_df = pd.DataFrame(data=final_data, columns=["judgement", "jec_class", "ano", "mes", "dia", "dia_semana", "juiz", "tipo_juiz", "indenizacao", "sentenca"])
    final_df.to_csv(PROCESSED_DATASET_W_SW, index=False)
