"""

"""
import glob

import pandas as pd
import tqdm

from util.path_constants import *
from util.value_contants import CLASSE_DICT, GET_INDIVIDUAL_VALUES


def merge_databases():
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
        attributes_df = pd.read_csv(DAMAGE_VALUES_DATASET_PATH, usecols=["Sentença", "Julgamento", "Valor individual do dano moral"])
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
    for index, row in attributes_df.iterrows():
        num_judgement = row["Sentença"]
        jec_class = row["Julgamento"]

        if GET_INDIVIDUAL_VALUES:
            indenizacao = float(str(row["Valor individual do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))
        else:
            indenizacao = float(str(row["Valor total do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))

        attributes_list.append([num_judgement, jec_class, indenizacao])

        for jec_judge in jec_list:
            # print(num_judgement, jec_class, indenizacao, jec_judge)

            if int(jec_judge[0]) == int(num_judgement) and \
                    jec_judge[1] == jec_class and \
                    (int(last_num) != int(num_judgement) or last_class != jec_class):
                final_data.append([int(num_judgement), jec_class, indenizacao, jec_judge[2]])

                break
        last_class = jec_class
        last_num = num_judgement

    final_df = pd.DataFrame(data=final_data, columns=["judgement", "jec_class", "indenizacao", "sentenca"])
    final_df.to_csv(MERGE_DATASET)
    # print(num_judgement, jec_class, indenizacao)
