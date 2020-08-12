# CS 677 Final Project

### NLP: English And Dutch Text Classifier

This project contains files for a machine learning program to predict the
 language of text messages as either Dutch or English. Before running the
  program, ensure following dependencies are installed:
  
- pandas
- sklearn
- matplotlib

### How to run

Each of the following files has a specific purpose and can be run separately to
  see the output in the terminal. Run using either command line or an IDE:
 
`main.py` is the main execution file. Run this file to see 2 example messages
 being classified (1 Dutch and 1 English). Then it will ask for input to classify a message string provided by the
   user.
  
`explore.py` is for exploratory analysis of the dataset. Run this file to see
 a sample of the dataset, and to save a histogram plot of message length by
  language to `txt_len_hist.png`.
  
`model.py` contains the machine learning model used to classify text messages
. Run this file to see performance metrics for both Naive Bayesian and Random
 Forest classifiers.