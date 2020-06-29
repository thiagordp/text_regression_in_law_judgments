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

#
# Merge Dataset
#
MERGE_DATASET = "data/merge_dataset.csv"

PROCESSED_DATASET = "data/processed_dataset.csv"
