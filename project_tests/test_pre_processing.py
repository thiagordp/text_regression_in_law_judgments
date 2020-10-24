"""

"""

from pre_processing import unify_databases


def test_unify_database():
    unify_databases.create_final_dataset(remove_stopwords=False, stemming=True)
    unify_databases.create_final_dataset(remove_stopwords=False, stemming=False)
    unify_databases.create_final_dataset(remove_stopwords=True, stemming=True)
    unify_databases.create_final_dataset(remove_stopwords=True, stemming=False)
