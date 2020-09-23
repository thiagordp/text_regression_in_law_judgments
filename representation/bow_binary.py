"""
@description
    Binary BOW
@author Thiago Raulino Dal Pont
"""

import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


def document_vector(corpus, dim=0, remove_stopwords=True, stemming=False):
    print("Binary BOW Document Vector")

    time.sleep(0.2)
    processed_corpus = list(corpus)

    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 4))
    x = vectorizer.fit_transform(processed_corpus).toarray()
    feature_names = vectorizer.get_feature_names()

    if dim > 0:
        print("Applying PCA to reduce to", dim, "dimensions")
        pca = PCA(n_components=dim)
        x = pca.fit_transform(x)
        print("Variance:", np.sum(pca.explained_variance_ratio_))

    return x, feature_names
