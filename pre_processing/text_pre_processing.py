"""

@author Thiago Raulino Dal Pont
"""
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('rslp')


def _clear_text(text):
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace("http://www ", " ")

    text = re.sub("-+", " ", text)
    text = re.sub("\.+", " ", text)
    text = text.replace("nbsp", " ")

    # Symbols

    for symb in "()[]{}!?\"§_/,-“”‘’–'º•|<>$#*@:;":
        text = text.replace(symb, " ")

    for symb in ".§ºª°":
        text = text.replace(symb, " ")

    for letter in "bcdfghjklmnpqrstvwxyz":
        text = text.replace(" " + letter + " ", " ")

    for i in range(20):
        text = text.replace("  ", " ")

    return text


def _remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if word not in stopwords.words('portuguese')]
    text = " ".join(tokens_without_sw)

    return text


def _stemming(text):
    text_tokens = word_tokenize(text)
    stemmer = nltk.stem.RSLPStemmer()

    text_tokens = [stemmer.stem(token) for token in text_tokens]
    text = " ".join(text_tokens)

    return text


def process_text(text, remove_stopwords=True, stemming=False):
    text = str(text)
    text = _clear_text(text)

    if remove_stopwords:
        text = _remove_stopwords(text)

    if stemming:
        text = _stemming(text)

    return text


def process_judge(judges, type_judges):
    judges = [str(judge_name).strip().lower().replace(" ", "_").replace(".", "") for judge_name in judges]
    type_judges = [str(judge_type).strip().lower().replace(" ", "_").replace(".", "") for judge_type in type_judges]

    print(sorted(set(judges)))

    judges = pd.get_dummies(judges, prefix='juiz')
    type_judges = pd.get_dummies(type_judges, prefix="tipo_juiz")

    judges = [list(judge) for judge in list(judges.to_numpy())]
    type_judges = [list(type_judge) for type_judge in list(type_judges.to_numpy())]

    return judges, type_judges


def process_has_x(feature=np.array([])):
    lb_make = LabelEncoder()

    return lb_make.fit_transform(feature)


def process_loss(feature=np.array([])):
    feature = [float(str(num).replace("-", "0").replace(",", ".")) for num in feature]

    return feature


def process_time_delay(feature=np.array([])):

    delay_minutes = list()

    for time_delay in feature:
        time_delay = time_delay.replace("- (superior a 4)", "00:00:00")
        time_delay = time_delay.replace("-", "00:00:00")
        splits = time_delay.split(":")

        seconds = float(splits[-1].strip()) / 60
        minutes = float(splits[-2].strip())
        hours = float(splits[-3].strip()) * 60

        delay_minutes.append(hours + minutes + seconds)

    return np.array(delay_minutes)
