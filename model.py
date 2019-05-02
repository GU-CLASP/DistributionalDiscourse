"""
Model classes for neural Dialogue Act Recognition.

There are two types of models: 
    - Encoders take an utterances and produce an n-dimensional vector.
    - DAR models, which take a sequence of encoded utterances (a conversation) 
      and produce a sequence of dialogue act tags.

Contents:
    - WordVecAvg - baseline word vector averaging 
    - DARRNN - a simple RNN DAR model
"""

import torch.nn as nn

class DARRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, utt_emsize, nlabels, nhid, nlayers, dropout=0.5):
        """
        ntokens - word vocabulary size # this should go in the encoder eventually
        nlabels - number of dialogue act tags 
        nembed  - size of the utterance embedding TODO: decouple from word embedding size
        nhid    - size of the hidden layer
        nlayers - number of hidden RNN layers
        """
        super(DARRNN, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.rnn = nn.RNN(utt_emsize, nhid, nlayers, nonlinearity='relu', dropout=dropout) # TODO: try tanh too?
        self.decoder = nn.Linear(nhid, nlabels)
        print(nlabels)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.drop(x)
        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        return decoded.view(x.size(0), x.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, batch_size, self.nhid)


class WordVecAvg(nn.Module):
    """ Baseline word vector encoder. Simply averages an utterance's word vectors
    """

    def __init__(self, wordvec_weights):
        super(WordVecAvg, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_vectors)

    def forward(self, x):
        return self.embedding(x).mean(dim=0)
