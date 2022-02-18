"""
File Path Constants
@author Thiago Raulino Dal Pont
"""

PROJECT_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Projects/text_regression_in_law_judgments/"

#
# Original Judgements
#

# TODO: Update this path to point to the downloaded dataset
JEC_DATASET_PATH = "/media/trdp/Arquivos/Studies/dev_msc/Datasets/processos_transp_aereo/txts_atualizados_sd_manual/"
# Classes
PROCEDENTE = "procedente/"
IMPROCEDENTE = "improcedente/"
PARC_PROCEDENTE = "parcialmente_procedente/"
EXTINCAO = "extincao/"

# JEC_CLASS_PATHS = [
#     PROCEDENTE,
#     IMPROCEDENTE,
#     PARC_PROCEDENTE,
#     EXTINCAO
# ]

# TODO: Place the dataset inside a folder "novos" (create it) in the path  JEC_DATA_SET.
JEC_CLASS_PATHS = [
    "novos/"
]

#
# Moral Damage Values Dataset
#
# TODO: create a folder data and place the attributes CSV file inside such folder.
DAMAGE_VALUES_DATASET_PATH = "data/regression_attributes_classes.csv"
INCLUDE_ZERO_VALUES = False

#
# Merge Dataset
#
MERGE_DATASET = "data/merge_dataset.csv"

PROCESSED_DATASET = "data/processed_dataset.csv"
PROCESSED_BASE_DATASET = "data/processed_dataset_@stopwords_@stemming.csv"
PROCESSED_DATASET_WO_STOP = "data/processed_dataset_w_sw.csv"
PROCESSED_DATASET_W_STOP = "data/processed_dataset_w_stop.csv"

# EMBEDDINGS_PATH = "/media/egov/Acer/Experiments/Datasets/law_embeddings_database/embeddings/air_transport/word2vec_cbow_10000000_100.txt"
# EMBEDDINGS_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/law_embeddings_database/embeddings/general/glove_3500000000_100.txt"

EMBEDDINGS_BASE_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/law_embeddings_database/embeddings/"
# EMBEDDINGS_BASE_PATH = "/media/egov/Acer/Experiments/Datasets/law_embeddings_database/embeddings/"
# EMBEDDINGS_BASE_PATH = "/media/egov/SSD_Files/Experiments/Datasets/law_embeddings_database/embeddings/"
EMBEDDINGS_LIST = [
    "air_transport/word2vec_cbow_100000000_100.txt",
    "air_transport/word2vec_sg_100000000_100.txt",
    "air_transport/glove_100000000_100.txt.txt",
    "air_transport/fasttext_cbow_100000000_100.txt",
    "air_transport/fasttext_sg_100000000_100.txt",
    # "general/glove_3500000000_100.txt.txt",
    # "general/fasttext_sg_3500000000_100.txt",
    # "general/fasttext_cbow_3500000000_100.txt",
    # "general/word2vec_sg_3500000000_100.txt",
    # "general/word2vec_cbow_3500000000_100.txt",
]

# ============== Contants for log evaluation ============== #
PATH_LOGS = "data/regression_logs/"
REPRESENTATIONS = [
    "embeddings/",
    "tf/",
    "tf_idf/"
]
