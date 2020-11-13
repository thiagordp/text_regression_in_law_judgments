"""

"""
import glob
import re
from datetime import datetime

import pandas as pd
import tqdm

from pre_processing.text_pre_processing import process_text
from util.path_constants import *
from util.value_contants import CLASSE_DICT, GET_INDIVIDUAL_VALUES


def create_final_dataset(remove_stopwords, stemming):
    final_data = list()
    attributes_list = list()
    jec_list = list()

    print("-" * 65)
    print("Loading JEC judgements")
    print("Remove Stopwords:", remove_stopwords)
    print("Apply Stemming:  ", stemming)

    for jec_class in JEC_CLASS_PATHS:
        class_path = JEC_DATASET_PATH + jec_class
        judgements_file_list = glob.glob(class_path + "*.txt")

        list_texts = dict()

        for judgement_file in tqdm.tqdm(judgements_file_list):
            text = open(judgement_file).read()
            text = text.replace("\n", " ")
            text = text.replace("\b", " ")
            text = text.replace("  ", " ")
            text = text.replace("  ", " ")

            file_name = judgement_file.replace(JEC_DATASET_PATH, "").replace(jec_class, "").replace(".txt", "").replace("/", "")
            jec_class = jec_class.replace("/", "")
            jec_list.append([file_name, jec_class, text])

            # hash_code = hashlib.sha3_512(str.encode(text)).hexdigest()
            #
            # if hash_code in list_texts:
            #     list_texts[hash_code].append(file_name)
            # else:
            #     list_texts[hash_code] = [file_name]
            # print(file_name, jec_class, text)

    print("Loadings attributes file")

    if GET_INDIVIDUAL_VALUES:
        attributes_df = pd.read_csv(DAMAGE_VALUES_DATASET_PATH,
                                    usecols=[
                                        "Sentença",
                                        "Julgamento",
                                        "Valor individual do dano moral",
                                        "Data do Julgamento",
                                        "Julgador(a)",
                                        "Tipo Julgador(a)",
                                        "Extravio Definitivo",
                                        "Extravio Temporário",
                                        "Intervalo do Extravio (dias)",
                                        "Violação (furto, avaria)",
                                        "Cancelamento (sem realocação)/Alteração de destino",
                                        "Atraso (com realocação)",
                                        "Intervalo do Atraso (horas:minutos)",
                                        "Culpa exclusiva do consumidor",
                                        "Condições climáticas desfavoráveis/Fechamento aeroporto",
                                        "No Show",
                                        "Overbooking",
                                        "Cancelamento pelo consumidor e problemas com o reembolso",
                                        "Descumprimento de oferta (assento)",
                                    ])
        # Remove rows which has one of the collumns with Null values
        attributes_df.dropna(subset=["Julgamento"], inplace=True)
        attributes_df.dropna(subset=["Atraso (com realocação)"], inplace=True)

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

        has_permanent_loss = row["Extravio Definitivo"]
        has_temporally_loss = row["Extravio Temporário"]
        interval_loss = row["Intervalo do Extravio (dias)"]
        has_luggage_violation = row["Violação (furto, avaria)"]
        has_flight_cancellation = row["Cancelamento (sem realocação)/Alteração de destino"]
        has_flight_delay = row["Atraso (com realocação)"]
        flight_delay = row["Intervalo do Atraso (horas:minutos)"]
        is_consumers_fault = row["Culpa exclusiva do consumidor"]
        has_adverse_flight_conditions = row["Condições climáticas desfavoráveis/Fechamento aeroporto"]
        has_no_show = row["No Show"]
        has_overbooking = row["Overbooking"]
        has_cancel_user_refunding_problem = row["Cancelamento pelo consumidor e problemas com o reembolso"]
        has_offer_disagreement = row["Descumprimento de oferta (assento)"]

        if GET_INDIVIDUAL_VALUES:
            indenizacao = float(str(row["Valor individual do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))
        else:
            indenizacao = float(str(row["Valor total do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))

        # attributes_list.append([num_judgement, jec_class, indenizacao])

        for jec_judge in jec_list:
            # print(num_judgement, jec_class, indenizacao, jec_judge)
            int_n = int(num_judgement)
            if int(jec_judge[0]) == int_n:  # and int_n != last_num:
                # jec_judge[1] == jec_class and \
                # (int(last_num) != int(num_judgement) or last_class != jec_class):
                processed_text = process_text(jec_judge[2], remove_stopwords=remove_stopwords, stemming=stemming)
                final_data.append([
                    int(num_judgement),
                    jec_class,
                    year,
                    month,
                    day,
                    weekday,
                    judge,
                    type_judge,
                    indenizacao,
                    has_permanent_loss,
                    has_temporally_loss,
                    interval_loss,
                    has_luggage_violation,
                    has_flight_delay,
                    has_flight_cancellation,
                    flight_delay,
                    is_consumers_fault,
                    has_adverse_flight_conditions,
                    has_no_show,
                    has_overbooking,
                    has_cancel_user_refunding_problem,
                    has_offer_disagreement,
                    re.sub(r'\W+', ' ', processed_text).replace("  ", " ").replace("  ", " ").strip()
                ])

                break

        last_class = jec_class
        last_num = num_judgement

    final_df = pd.DataFrame(data=final_data,
                            columns=[
                                "judgement",
                                "jec_class",
                                "ano",
                                "mes",
                                "dia",
                                "dia_semana",
                                "juiz",
                                "tipo_juiz",
                                "indenizacao",
                                "extravio_permanente",
                                "extravio_temporario",
                                "intevalo_extravio",
                                "tem_violacao_bagagem",
                                "tem_atraso_voo",
                                "tem_cancelamento_voo",
                                "qtd_atraso_voo",
                                "culpa_consumidor",
                                "tem_condicao_adversa_voo",
                                "tem_no_show",
                                "tem_overbooking",
                                "tem_cancelamento_usuario_ressarcimento",
                                "tem_desacordo_oferta",
                                "sentenca"
                            ])

    file_path = PROCESSED_BASE_DATASET
    if remove_stopwords:
        file_path = file_path.replace("@stopwords", "wo_stopwords")
    else:
        file_path = file_path.replace("@stopwords", "w_stopwords")

    if stemming:
        file_path = file_path.replace("@stemming", "w_stemming")
    else:
        file_path = file_path.replace("@stemming", "wo_stemming")

    final_df.to_csv(file_path, index=False)
