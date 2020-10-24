"""

"""
import glob
import re

import nltk
from sklearn.feature_extraction.text import CountVectorizer

from util.path_constants import JEC_DATASET_PATH

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" -- ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def jec_lda():
    files = glob.glob(JEC_DATASET_PATH + "novos/*.txt")
    texts = list()

    stopwords = nltk.corpus.stopwords.words('portuguese')

    for f in files:
        text = open(f).read()
        text = re.sub('[,.!?]', '', text)
        text = text.lower()
        tokens = [x for x in text.split() if x not in stopwords]
        texts.append(" ".join(tokens))

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer()  # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(texts)

    # Tweak the two parameters below
    number_topics = 40
    number_words = 5  # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)  # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)
