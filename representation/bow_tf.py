"""

@author Thiago Raulino Dal Pont
"""
import time

import nltk
import tqdm
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from pre_processing.text_pre_processing import process_text


def document_vector(corpus, dim=0, remove_stopwords=True, stemming=False):
    print("TF BOW Document Vector")
    time.sleep(0.2)
    processed_corpus = list(corpus)
    # for doc in tqdm.tqdm(corpus):
    #     proc = process_text(doc, remove_stopwords=remove_stopwords, stemming=stemming)
    #     processed_corpus.append(proc)

    vectorizer = CountVectorizer(ngram_range=(1, 4))
    x = vectorizer.fit_transform(processed_corpus).toarray()
    feature_names = vectorizer.get_feature_names()

    if dim > 0:
        print("Applying PCA to reduce to", dim, "dimensions")
        pca = PCA(n_components=dim)
        x = pca.fit_transform(x)
        print("Variance:", np.sum(pca.explained_variance_ratio_))

    return x, feature_names
