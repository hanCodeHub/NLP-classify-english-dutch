"""
Created: August 10, 2020
@author: han10

Main execution program.
Inputs messages in English or Dutch into ML model.
"""
from data import *
from model import predict_label
from sklearn.naive_bayes import MultinomialNB
import sys


def predict_language(message, classifier=MultinomialNB):
    """
    Wrapper function for predicting the language of given message string
    Precondition: x_train and y_train are pre-computed
    """
    pred = predict_label([message], classifier, x_train, y_train)[0]
    return 'English' if pred == 'en' else 'Dutch'


if __name__ == '__main__':
    try:
        # prepares data
        df_msg = read_data('raw_text.txt')
        x_train, x_test, y_train, y_test = split_data(df_msg)

        # attempts Dutch prediction
        nl_msg = 'Dit is mijn eerste keer dat ik NLP gebruik.'
        print(f'Predicting Dutch message: {nl_msg}')
        pred_nl = predict_language(nl_msg)
        print(f'Predicted language: {pred_nl}')

        print()

        # attempts English prediction
        en_msg = 'This is my first time using NLP.'
        print(f'Predicting English message: {en_msg}')
        pred_en = predict_language(en_msg)
        print(f'Predicted language: {pred_en}')

        # attempts prediction of user message
        print()
        prompt = 'Please provide a message in either English or Dutch:\n'
        user_msg = input(prompt)
        pred_user = predict_language(user_msg)
        print(f'Predicted language: {pred_user}')

    except Exception as e:
        print(f'could not execute script: {e}')
