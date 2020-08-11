"""
Created: August 10, 2020
@author: han10

Program for exploring the given dataset.
Outputs and plots the data to extract meaningful insight.
"""
import matplotlib.pyplot as plt

from data import *

if __name__ == '__main__':
    try:
        # reads and outputs head of dataframe to terminal
        df_msg = read_data('raw_text.txt')
        print('Original dataframe:')
        print(df_msg)

        print()

        # checks number of instances for each language
        print(f'Number of Dutch messages: {sum(df_msg["label"] == "nl")}')
        print(f'Number of English messages: {sum(df_msg["label"] == "en")}')

        print()

        # explores difference in length between en and nl text
        df_msg['length'] = df_msg['message'].apply(len)
        # plots text length histogram
        print('Plotting histogram of text length by text language...')
        fig = plt.figure()
        df_msg.hist(column='length', by='label', bins=50, figsize=(12, 4))
        plt.savefig('txt_len_hist.png')
        print('Saved plot to txt_len_hist.png')

    except Exception as e:
        print(f'could not execute script: {e}')
