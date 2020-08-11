"""
Created: August 10, 2020
@author: han10

Provides utility functions for reading and processing data.
"""

import pandas as pd
import string
from sklearn.model_selection import train_test_split


def read_data(filename):
    """reads given txt file into a pandas dataframe and returns it"""
    return pd.read_csv(filename, sep='|', names=['label', 'message'])


def split_data(df):
    """
    splits data and returns training and testing sets for x and y
    return in the order of: x_train, x_test, y_train, y_test
    """
    feature, target = df['message'], df['label']
    return train_test_split(feature, target, test_size=0.5, random_state=101)


def clean_message(msg):
    """removes punctuation from given msg and returns a list of its words"""
    clean_msg = [c for c in msg if c not in string.punctuation]
    clean_msg = ''.join(clean_msg)
    clean_msg_arr = clean_msg.split(' ')
    return [x for x in clean_msg_arr if x.isalpha()]  # exclude numbers/symbols
