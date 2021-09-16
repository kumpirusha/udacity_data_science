import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize


class TextLengthExtractor:

    def __init__(self, base_estimator, transformer_matrix):
        self.base_estimator = base_estimator
        self.transformer_matrix = transformer_matrix

    def text_len(self, text):
        """
        A simple function that returns the length of a text.
        :param: A string of text
        :returns: Length
        """
        length = len(word_tokenize(text))
        return length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        INPUT:
            X - A NumPy array or pd series, contains a series of messages used for classification
        OUTPUT:
            pd.DataFrame(X_length) - A panda data frame, - contains the number of tokens in the text
        """
        X_length = pd.Series(X).apply(self.get_length)

        return pd.DataFrame(X_length)
