# ELMO Model Implementation using Bidirectional LSTMs
This repository contains the implementation of an ELMO (Embeddings from Language Models) style model using PyTorch. ELMO is a deep contextualized word representation model that captures both word-level and contextual information from text.

## Model Architecture 
* LSTM Class: Defines the ELMO model using two stacked bidirectional LSTMs:
* Embedding layer initialized with pre-trained word embeddings.
* Two LSTM layers with bidirectional processing.
* Linear layer for predicting word indices.

## Dataset
The dataset used in this implementation is derived from the Google News dataset. I have utilized word embeddings from the "word2vec-google-news-300" model. This dataset consists of word vectors pre-trained on a large corpus of Google News articles, where each word is represented as a 300-dimensional vector.

## Execution
1. python elmo.py
-- This will generate two dictionaries word2idx.pt and idx2word.pt. These dictionaries will be used further for the downstream classification task. 
-- Only 20000 sentences from the train corpus have been used for generating the dictionaries as well as for training.
-- After training, two pretrained models named forward_model.pt and backward_model.pt are generated (instead of a single bilstm.pt)

2. python classification.py
-- This will generate the classifier.pt file.

## Hyper-Parameter Tuning
The hyperparameter tuning part consists of three parts:
* Trainable lambdas
* Frozen lambdas
* Learnable function

These parts are implemented in the HyperparamTuning.ipynb.
