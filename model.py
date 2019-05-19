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
from pytorch_pretrained_bert import BertModel 

class DARRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, utt_size, n_tags, hidden_size, n_layers, dropout=0.5):
        """
        utt_size    - size of the encoded utterance 
        hidden_size - size of the hidden layer 
        n_tags      - number of dialogue act tags 
        n_layers    - number of hidden RNN layers
        """
        super().__init__()
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
        super().__init__()
        self.embedding = embedding 

    @classmethod
    def from_pretrained(cls, weights):
        embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
        return cls(embedding)

    @classmethod
    def random_init(cls, num_embeddings, embedding_dim):
        embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        return cls(embedding)

    def forward(self, x, x_lens):
        x = self.embedding(x).sum(dim=1) / x_lens.unsqueeze(1) 
        return x 

class BertUttEncoder(BertModel):

    @classmethod
    def from_pretrained_base_uncased(cls):
        return super().from_pretrained('bert-base-uncased')

    def forward(self, x, x_lens):
        hidden_states, pooled_output = super().forward(x)
        return pooled_output
