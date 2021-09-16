# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import pickle
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, \
    recall_score, accuracy_score, f1_score, make_scorer
from custom_transformer import TextLengthExtractor


# load data from database
def load_data(db_name, table):
    """
    Function that loads the data from the local SQL database.
    :param db_name: Name of the connection database
    :param table: Name of the table inside the 'db_name' database
    :return: Two pandas dataframes X,y containing the disaster reponse 'messages' and 'categories' and a
    list containing 'y' df column names
    """
    engine = create_engine('sqlite:///{database}.db'.format(database=db_name))
    df = pd.read_sql_table(table_name=table, con=engine)

    X = df.message
    y = df.iloc[:, 4:]
    cats = y.columns

    return X, y, cats


# tokenize text
def tokenize(text):
    """
    The function cleans the input text string and splits is into a list of keywords
    :param text: A string text
    :return: A list of strings
    """
    text_clean = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text_tokens = word_tokenize(text_clean)
    no_sw_tokens = [i for i in text_tokens if i not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_text = []
    for token in no_sw_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_text.append(clean_token)

    return clean_text


def build_model():
    """
    Function used to build the ML model for disaster text evaluation.
    :return: A grid-search pipeline used to train the model and find optimize parameter selection
    """
    # Create an ML pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ]))
            ('text_length', TextLengthExtractor())
        ])),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(random_state=15)))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__max_depth': [15, 30],
        'clf__estimator__n_estimators': [100, 250],
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False)
    }

    # Use grid search to find better parameters.
    GridSearchCV_pipeline = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)

    return GridSearchCV_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model on the test data and print the result
    input:
        model  = trained classifier model
        X_test = testing features (Unseen data)
        Y_test = true values to compare with prediction on unseen test cases
        category_names = column name of Y_test data
    output:
        print model prediction accuracy on test data
    """
    y_pred = model.predict(X_test)
    metrics_list_all = []
    for col in range(Y_test.shape[1]):
        accuracy = accuracy_score(Y_test.iloc[:, col], y_pred[:, col])
        precision = precision_score(Y_test.iloc[:, col], y_pred[:, col])
        recall = recall_score(Y_test.iloc[:, col], y_pred[:, col])
        f_1 = f1_score(Y_test.iloc[:, col], y_pred[:, col])
        metrics_list = [accuracy, precision, recall, f_1]
        metrics_list_all.append(metrics_list)
    metrics_df = pd.DataFrame(metrics_list_all,
                              index=category_names,
                              columns=["Accuracy", "Precision", "Recall", "F_1"])
    print(metrics_df)


def save_model(model, model_filepath):
    """
    Saving trained model on on disk to be load when required.
    input:
        model = trained classifier
        model_filepath = file path and name to store model on disk

    """
    # using pickle to store trained classifier
    with gzip.open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
