import torch.nn as nn

class DARRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, word_embedding, nembed, nlabels, nhid, nlayers, dropout=0.5):
        """
        ntokens - word vocabulary size # this should go in the encoder eventually
        nlabels - number of dialogue act tags 
        nembed  - size of the utterance embedding TODO: decouple from word embedding size
        nhid    - size of the hidden layer
        nlayers - number of hidden RNN layers
        """
        super(DARRNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_embedding = word_embedding 

        self.rnn = nn.RNN(nembed, nhid, nlayers, nonlinearity='relu', dropout=dropout) # TODO: try tanh too?
        self.decoder = nn.Linear(nhid, nlabels)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, utt, hidden):
        # emb = self.drop(self.encoder(input))
        utt_embd = self.word_embedding(utt).mean(dim=0)
        output, hidden = self.rnn(utt_embd, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, batch_size, self.nhid)
