"""
VSM using mean of embeddings

@author Thiago R. Dal Pont
@date Oct 5, 2020
"""
import random
import time

import tqdm
import numpy as np
from gensim.models import KeyedVectors

from util.path_constants import EMBEDDINGS_BASE_PATH, EMBEDDINGS_LIST


def average_arrays(list_arrays):

    x = 0

def document_vector(corpus, dim=0, embeddings_path=EMBEDDINGS_BASE_PATH + EMBEDDINGS_LIST[0]):
    print("Mean Embeddings Document Vector")
    time.sleep(0.2)
    processed_corpus = list(corpus)
    # for doc in tqdm.tqdm(corpus):
    #     proc = process_text(doc, remove_stopwords=remove_stopwords, stemming=stemming)
    #     processed_corpus.append(proc)

    word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

    corpus_embeddings = list()

    random_array = [(random.random() * 2 - 1) for i in range(100)]
    for doc in tqdm.tqdm(corpus):
        doc_emb = list()
        tokens = doc.split()

        for token in tokens:
            try:
                emb_token = word_vectors[token]
            except:
                emb_token = random_array

            doc_emb.append(emb_token)

        doc_emb = np.sum(doc_emb, axis=0)
        corpus_embeddings.append(doc_emb)

    return corpus_embeddings, []
