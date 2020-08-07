"""
Regression models using word embeddings
"""
import tqdm
from keras import backend, regularizers, Model, metrics, Input
from keras.layers import concatenate, Flatten, Dropout, Dense, Reshape, Embedding, Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer

import numpy as np

from representation import word_embeddings


def simple_cnn_model(x_train, y_train, x_test, y_test, x_val, y_val, vocabulary_size, emb_len, embedding_matrix, embeddings_path):
    results_list = list()

    for i in tqdm.tqdm(range(20)):
        backend.clear_session()

        # Define Embedding function using the embedding_matrix
        embedding_layer = Embedding(vocabulary_size, emb_len, weights=[embedding_matrix], trainable=False)

        sequence_length = x_train.shape[1]
        filter_sizes = [2, 3, 4, 5]
        num_filters = 10
        drop = 0.5

        inputs = Input(shape=(sequence_length,))
        embedding = embedding_layer(inputs)
        reshape = Reshape((sequence_length, emb_len, 1))(embedding)

        convs = []
        maxpools = []

        for filter_size in filter_sizes:
            conv = Conv2D(num_filters, (filter_size, emb_len), activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)

            maxpool = MaxPooling2D((sequence_length - filter_size + 1, 1), strides=(1, 1))(conv)

            maxpools.append(maxpool)
            convs.append(conv)

        merged_tensor = concatenate(maxpools, axis=1)

        flatten = Flatten()(merged_tensor)
        # reshape = Reshape((3 * num_filters,))(flatten)
        dropout = Dropout(drop)(flatten)
        output = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01))(dropout)

        # Build model
        model = Model(inputs, output)
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit model
        hist_adam = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0, validation_data=(x_val, y_val), shuffle=True)

        # Predict for test set
        y_pred = model.predict(x_test)

        results_list.append([embeddings_path, y_test, y_pred])

    # Return predictions
    return results_list
