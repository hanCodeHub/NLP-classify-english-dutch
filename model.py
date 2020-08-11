"""
Created: August 10, 2020
@author: han10

Program for modeling the dataset and training a classifier.
Predicts the language label ('en' or 'nl') when given a text snippet.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from data import *


def predict_label(msg_arr, classifier, x_train, y_train):
    """
    Returns predictions on given array of messages
    Given classifier and training data
    """
    # passes a list of steps into sklearn pipeline
    pipeline = Pipeline([
        # vectorizes tokens into numerical data using bag-of-words model
        ('vectorizer', CountVectorizer(analyzer=clean_message)),
        # computes Term Frequency - Inverse Document Frequency
        ('tfidf', TfidfTransformer()),
        # trains the model with given classifier
        ('classifier', classifier())
    ])

    # applies all pipeline steps to the given datasets and returns predictions
    pipeline.fit(x_train, y_train)
    return pipeline.predict(msg_arr)


def output_results(y_test, predictions):
    """Outputs metrics to terminal based on given test data and predictions"""
    accuracy = accuracy_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f'Accuracy: {round(accuracy * 100, 2)}%')
    print(f'True Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'True Negatives: {tn}')
    print(f'True Positive Rate: {tp / (tp + fn)}')
    print(f'True Negative Rate: {tn / (tn + fp)}')


if __name__ == '__main__':
    try:
        df_msg = read_data('raw_text.txt')

        # splits data between train and test
        x_train, x_test, y_train, y_test = split_data(df_msg)

        # predicts with Naive Bayes classifier and output results to terminal
        nb_pred = predict_label(x_test, MultinomialNB, x_train, y_train)
        print('Results for Naive Bayes classifier:')
        output_results(y_test, nb_pred)

        print('\n')

        # predicts with Random Forest classifier and output results to terminal
        rf_pred = predict_label(x_test, RandomForestClassifier,
                                x_train, y_train)
        print('Results for Random Forest classifier:')
        output_results(y_test, rf_pred)

    except Exception as e:
        print(f'could not execute script: {e}')
