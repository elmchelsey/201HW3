# 201HW3

## Running the code

Simply run `python main.py` to train the HMM and evaluate the performance of the model on the test set with the best alpha value found by the tuning process.



# Changes to starter code

1. I added a return statement to the evaluate function so that the accuracy score is returned in addition to the print statement that was already there (I did this so I could use the accuracy score in the main function).

2. I added a print argument to the evaluate function that is set to False by default so that the accuracy score is not printed unless set to True.

## Overview

This project implements a Hidden Markov Model for Part-of-Speech tagging using:
- Transition probabilities between tags
- Emission probabilities from tags to words
- Viterbi algorithm for decoding
- Laplace (additive) smoothing

## Features

- HMM-based POS tagging with Viterbi decoding
- Laplace smoothing with tunable alpha parameter
- Confusion matrix visualization
- Performance optimization using NumPy
- Error analysis capabilities
- Model evaluation tools (starter_code.py provided by instructor)