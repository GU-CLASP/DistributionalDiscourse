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

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class SimpleDARRNN(nn.Module):
    def __init__(self, utt_size, n_tags):
        super().__init__()
        self.decoder = nn.Linear(utt_size, n_tags)
    def forward(self,x,hidden):
        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        return decoded.view(x.size(0), x.size(1), decoded.size(1)), hidden
    def init_hidden(self, batch_size):
        import torch
        return torch.tensor([])


class DARRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, utt_size, n_tags, hidden_size, n_layers, dropout=0.5, use_lstm=False):
        """
        utt_size    - size of the encoded utterance 
        hidden_size - size of the hidden layer 
        n_tags      - number of dialogue act tags 
        n_layers    - number of hidden RNN layers
        """
        super().__init__()
        self.use_lstm = use_lstm
        self.drop = nn.Dropout(dropout)

        if use_lstm:
            self.rnn = nn.LSTM(utt_size, hidden_size, n_layers, dropout=dropout)
        else:
            self.rnn = nn.RNN(utt_size, hidden_size, n_layers, nonlinearity='relu', dropout=dropout) # TODO: try tanh too?
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
        if self.use_lstm:
            return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
        return weight.new_zeros(self.n_layers, batch_size, self.hidden_size)


class WordVecAvg(nn.Module):
    """ Baseline word vector encoder. Simply averages an utterance's word vectors
    """

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding 

    @classmethod
    def from_pretrained(cls, weights):
        embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        return cls(embedding)

    @classmethod
    def random_init(cls, num_embeddings, embedding_dim):
        embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        return cls(embedding)

    def forward(self, x):
        x = self.embedding(x).sum(dim=1) 
        return x 

class KimCNN(nn.Module):
    """ CNN utt encoder based on Kim (2014): https://github.com/yoonkim/CNN_sentence
    """

    def __init__(self, vocab_size, utt_size, embedding_dim, embedding, window_sizes, feature_maps):
        super().__init__()

        self.embedding = embedding
        self.convs = nn.ModuleList([nn.Conv2d(1, feature_maps, (window_size, embedding_dim))
                                    for window_size in window_sizes])
        self.linear = nn.Linear(len(window_sizes)*feature_maps, utt_size)
        self.dropout = nn.Dropout(0.5)

    @classmethod
    def from_pretrained(cls, vocab_size, utt_size, embedding_dim, weights, *args):
        embedding = nn.Embedding.from_pretrained(weights, freeze=freeze_embedding, padding_idx=0)
        return cls(vocab_size, utt_size, embedding_dim, embedding, *args)

    @classmethod
    def random_init(cls, vocab_size, utt_size, embedding_dim, *args):
        embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        return cls(vocab_size, utt_size, embedding_dim, embedding, *args)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class BertUttEncoder(nn.Module):

    def __init__(self, utt_size, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            config = transformers.BertConfig.from_json_file('data/bert-base-uncased_config.json')
            self.bert = transformers.BertModel(config)
        self.linear = nn.Linear(768, utt_size)

    def forward(self, x):
        _, x = self.bert(x)  # use the pooled [CLS] token output (_ is the 12 hidden states)
        x = self.linear(x)
        return x
