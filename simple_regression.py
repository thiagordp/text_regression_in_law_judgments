"""
First regression test
"""
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVR

from pre_processing.text_pre_processing import process_text
from util.path_constants import MERGE_DATASET, PROCESSED_DATASET


def create_processed_base():
    print("Create processed base")

    # Import data
    raw_data_df = pd.read_csv(MERGE_DATASET, index_col=0)

    # Representation
    x = [row for row in raw_data_df["sentenca"].values]

    y = raw_data_df["indenizacao_total"].values

    print("Processing texts")
    data = list()
    for i in tqdm.tqdm(range(len(x))):
        row = x[i]
        label = y[i]
        line = process_text(row, remove_stopwords=True, stemming=True)
        data.append([label, line])

    new_df = pd.DataFrame(data, columns=["values", "text"])
    new_df.to_csv(PROCESSED_DATASET)


def simple_regression():
    print("Simple Regression")

    # Import data
    raw_data_df = pd.read_csv(PROCESSED_DATASET, index_col=0)
    raw_data_df.dropna(inplace=True)
    raw_data_df["values"] = raw_data_df["values"].fillna(value=0)
    print("\tRead", len(raw_data_df), "rows")

    x = raw_data_df["text"].values
    y = raw_data_df["values"].values

    # Representation
    print("TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    x = vectorizer.fit_transform(x).toarray()

    print(len(vectorizer.vocabulary_))

    scaler = MinMaxScaler(feature_range=(0, 1))
    y = y.reshape(-1, 1)
    # y = scaler.fit_transform(y)
    # print(scaler.data_max_)
    # print(scaler.data_min_)
    # Split train / test
    print("Extracting polinomial features")
    #polynomial_features = PolynomialFeatures(degree=4)
    #x = polynomial_features.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=int(time.time() % 2 ** 32))

    #for i in range(len(x)):
    #    print(len(x[i]), x[i])
    # Train Regressor
    # regressor = SVR(kernel='rbf', verbose=2, )
    # regressor.fit(x_train, y_train)

    print("PCA")
    pca = PCA(n_components=1)
    x_pca = pca.fit_transform(x_test)
    print(pca.explained_variance_ratio_)

    plt.scatter(x_pca, y_test)

    # regressor = LinearRegression()
    # regressor.fit(x_train, y_train)  # training the algorithm
    # Tests
    # y_pred_scalled = regressor.predict(x_test)
    # y_p = list()
    # y_t = list()
    # for i in range(len(y_pred_scalled)):
    #     y_p.append(y_test[i] * (scaler.data_max_ - scaler.data_min_))
    #     y_t.append(y_pred_scalled[i] * (scaler.data_max_ - scaler.data_min_))
    #
    # print(np.sqrt(metrics.mean_squared_error(y_t, y_p)))
    # print(metrics.r2_score(y_t, y_p))
    print("Regressor")
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_poly_pred = model.predict(x_test)

    plt.scatter(x_pca, y_poly_pred)
    plt.show()

    print("Evaluating")
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_poly_pred))
    r2 = metrics.r2_score(y_test, y_poly_pred)
    mae = metrics.mean_absolute_error(y_test, y_poly_pred)
    print(rmse)
    print(r2)
    print(mae)

    for i in range(len(y_test)):
        print(y_test[i], "\t", y_poly_pred[i])
        time.sleep(0.5)

    return None


def main():
    # create_processed_base()
    simple_regression()


if __name__ == "__main__":
    main()
