# Part of Speech (POS) Tagging using a Hidden Markov Model (HMM) with Viterbi Decoding

An HMM-based POS tagger trained using the Penn Tree Bank (PTB) tagset and the Wall Street Journal (WSJ) section of the PTB corpus. This project was completed as part of my coursework for UCSC's NLP MSc program.

## Running the code

Simply run `python main.py` to train the HMM and evaluate the performance of the model on the test set with the best alpha value found by the tuning process.

## Files

- `main.py`: the main file that contains the HMM implementation and the tuning process.
- `starter_code.py`: the starter code provided by the instructor. Used mainly to load the data and evaluate the models. The evaluate() function was modified to return the accuracy score.

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
