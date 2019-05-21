import unittest
import random
import itertools
import logging

import util
import model
import train

import torch.optim as optim
import torch.nn as nn


log = util.create_logger(logging.DEBUG)
train.log = log  

# synthetic training data
random.seed(777)
input_vocab = list(range(50)) # 50 token vocab
label_vocab = list(range(7))  # 7 DA tags
data = [[random.choices(input_vocab, k=random.randint(3,10)) # utts of 3-10 tokens
        for r in range(random.randint(2,15))] # 2-15 utts per dialogue
        for i in range(5)] # 5 dialogues
labels = [random.choices(label_vocab, k=len(convo)) for convo in data]

utt_dims = 250
n_hidden = 50
vocab_size = len(input_vocab)
n_labels = len(label_vocab)
train_data = list(zip(data, labels))
epochs = 2
batch_size = 2
bptt = 5
class TestTraining(unittest.TestCase):

    def test_wordvecavg(self):
        utt_encoder = model.WordVecAvg.random_init(vocab_size, utt_dims)
        dar_model = model.DARRNN(utt_dims, n_labels, n_hidden, 1, dropout=0)
        train_params = itertools.chain(dar_model.parameters(), utt_encoder.parameters())
        optimizer=optim.Adam(train_params)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  
        print("Testing Word2VecAvg on random inputs.")
        for epoch in range(epochs):
            train.train_epoch(utt_encoder, dar_model, train_data, n_labels, 
                    batch_size, bptt, None, criterion, optimizer, 'cpu')

    def test_bert(self):
        utt_encoder = model.BertUttEncoder(utt_dims)
        dar_model = model.DARRNN(utt_dims, n_labels, n_hidden, 1, dropout=0)
        train_params = itertools.chain(dar_model.parameters(), utt_encoder.parameters())
        optimizer=optim.Adam(train_params)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  
        print("Testing BERT on random inputs.")
        for epoch in range(epochs):
            train.train_epoch(utt_encoder, dar_model, train_data, n_labels, 
                    batch_size, bptt, None, criterion, optimizer, 'cpu')


