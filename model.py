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

    def __init__(self, utt_size, n_tags, hidden_size, n_layers, dropout=0.5):
        """
        utt_size    - size of the encoded utterance 
        hidden_size - size of the hidden layer 
        n_tags      - number of dialogue act tags 
        n_layers    - number of hidden RNN layers
        """
        super(DARRNN, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.rnn = nn.RNN(utt_size, hidden_size, n_layers, 
                nonlinearity='relu', dropout=dropout) # TODO: try tanh too?
        self.decoder = nn.Linear(hidden_size, n_tags)

        self.init_weights()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

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
        return weight.new_zeros(self.n_layers, batch_size, self.hidden_size)


class WordVecAvg(nn.Module):
    """ Baseline word vector encoder. Simply averages an utterance's word vectors
    """

    def __init__(self, embedding):
        super(WordVecAvg, self).__init__()
        self.embedding = embedding 

    def forward(self, x):
        return self.embedding(x).mean(dim=0)
