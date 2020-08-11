"""
Created: August 10, 2020
@author: han10

Provides utility functions for reading and processing data.
"""

import pandas as pd
import string


def read_data(filename):
    """reads given txt file into a pandas dataframe and returns it"""
    return pd.read_csv(filename, sep='|', names=['label', 'message'])


def clean_message(msg):
    """removes punctuation from given msg and returns a list of its words"""
    clean_msg = [c for c in msg if c not in string.punctuation]
    clean_msg = ''.join(clean_msg)
    clean_msg_arr = clean_msg.split(' ')
    return [x for x in clean_msg_arr if x.isalpha()]  # exclude numbers/symbols
