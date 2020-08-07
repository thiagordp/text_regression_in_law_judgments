"""

@author Thiago Raulino Dal Pont
"""
import random

import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from pre_processing.text_pre_processing import process_text
from util.value_contants import MAX_NUMBER_WORDS, EMBEDDINGS_MAX_LEN_SEQ, EMBEDDINGS_LEN
import numpy as np


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])


def document_matrix(corpora, word_vectors):
    documents_matrices = list()

    # Process text
    processed_corpora = list()
    for doc in tqdm.tqdm(corpora):
        proc = process_text(doc, remove_stopwords=False, stemming=False)
        processed_corpora.append(proc)

    # Get embeddings
    tokenizer = Tokenizer(num_words=MAX_NUMBER_WORDS, filters='!"#$%&()*+,-./:;<=>?@–[\\]^_`{|}~\t\n\'', lower=True)
    tokenizer.fit_on_texts(corpora)
    word_index = tokenizer.word_index

    sequences_train = tokenizer.texts_to_sequences(corpora)
    x = pad_sequences(sequences_train, maxlen=EMBEDDINGS_MAX_LEN_SEQ)

    vocabulary_size = min(len(word_index) + 1, MAX_NUMBER_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDINGS_LEN))

    vec = [(random.random() * 2) - 1 for i in range(EMBEDDINGS_LEN)]
    
    # vec = np.random.rand(EMBEDDINGS_LEN)
    for word, i in word_index.items():
        if i >= MAX_NUMBER_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            vec = np.random.rand(EMBEDDINGS_LEN)
            embedding_matrix[i] = vec

    # simple_cnn_model(x_train, y_train, x_test, y_test, vocabulary_size, emb_len, embedding_matrix):

    return x, vocabulary_size, EMBEDDINGS_LEN, embedding_matrix
