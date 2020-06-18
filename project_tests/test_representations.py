"""

"""

from representation import bow_tf
from nltk.corpus import stopwords


def test_bow_tf():
    texts = [line.replace("\n", " ") for line in open("data/text_sample.txt")]

    bow_tf.document_vector(texts, dim=100, remove_stopwords=True)
