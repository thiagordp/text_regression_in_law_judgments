"""

@author Thiago Raulino Dal Pont
"""
import re

import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

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
