"""

@author Thiago Raulino Dal Pont
"""

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def document_vector(corpus, dim=100, remove_stopwords=True, lemmatization=False, stemming=False):
    print("TF BOW Document Vector")
    if remove_stopwords:
        docs = list(corpus)
        corpus = []

        for text in docs:
            tokens = [token for token in text.split() if token not in stopwords.words('portuguese')]
            corpus.append(" ".join(tokens))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    return X.toarray()
