"""
File Path Constants
@author Thiago Raulino Dal Pont
"""

#
# Original Judgements
#
JEC_DATASET_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/processos_transp_aereo/merge_sem_dispositivo/"
# Classes
PROCEDENTE = "procedente/"
IMPROCEDENTE = "improcedente/"
PARC_PROCEDENTE = "parcialmente_procedente/"
EXTINCAO = "extincao/"

JEC_CLASS_PATHS = [
    PROCEDENTE,
    IMPROCEDENTE,
    PARC_PROCEDENTE,
    EXTINCAO
]



#
# Moral Damage Values Dataset
#
DAMAGE_VALUES_DATASET_PATH = "data/regression_attributes_classes.csv"
INCLUDE_ZERO_VALUES = False

#
# Merge Dataset
#
MERGE_DATASET = "data/merge_dataset.csv"

PROCESSED_DATASET = "data/processed_dataset.csv"

# EMBEDDINGS_PATH = "/media/egov/Acer/Experiments/Datasets/law_embeddings_database/embeddings/air_transport/word2vec_cbow_10000000_100.txt"
# EMBEDDINGS_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/law_embeddings_database/embeddings/general/glove_3500000000_100.txt"

# EMBEDDINGS_BASE_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/law_embeddings_database/embeddings/"
# EMBEDDINGS_BASE_PATH = "/media/egov/Acer/Experiments/Datasets/law_embeddings_database/embeddings/"
EMBEDDINGS_BASE_PATH = "/media/egov/SSD_Files/Experiments/Datasets/law_embeddings_database/embeddings/"
EMBEDDINGS_LIST = [
    "air_transport/word2vec_cbow_10000000_100.txt",
    "air_transport/word2vec_sg_10000000_100.txt",
    "air_transport/glove_100000000_100.txt.txt",
    "air_transport/fasttext_cbow_10000000_100.txt",
    "air_transport/fasttext_sg_10000000_100.txt",
    "general/glove_3500000000_100.txt.txt",
    "general/fasttext_sg_3500000000_100.txt",
    "general/fasttext_cbow_3500000000_100.txt",
    "general/word2vec_sg_3500000000_100.txt",
    "general/word2vec_cbow_3500000000_100.txt",
]

# ============== Contants for log evaluation ============== #
PATH_LOGS = "data/regression_logs/"
REPRESENTATIONS = [
    "embeddings/",
    "tf/",
    "tf_idf/"
]
