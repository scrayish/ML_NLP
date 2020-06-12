# ML_NLP

This repository contains all sourcecode used in training models for Bachelor's thesis.
These models are RNNs for Quote Generation (Natural Language Processing)

Inside the folder "models" are 4 models used in thesis:
1) LSTM RNN using randomly generated embeddings
2) LSTM RNN using GloVe 6B (300d) embeddings
3) GRU RNN using randomly generated embeddings 
4) GRU RNN using GloVe 6B (300d) embeddings

Notes:
Random embeddings are initialized to 300 dimensions to be on par with GloVe
main.py contains main executable for training models
dataset.py initializes datasets
pre_processing.py contains data preprocessing, also supports opening already preprocessed data files, if they're preprocessed
by this same preprocessing function
utility.py contains utility functions used in preprocessing and other calculations
generator.py is a generator program, which can be used to generate quotes, instructions given on top of file
