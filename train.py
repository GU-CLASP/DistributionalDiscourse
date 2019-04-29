import torch
import torch.nn as nn
from model import DARRNN
import json 
import pprint 
from collections import Counter

pp = pprint.PrettyPrinter(indent=4)

vocab_limit = 5000
glove_dim = 50
nlayers = 1
nhidden = 200

# Initialize word embedding from glove vectors
# TODO: Use google news word2vec also/instead?
# TODO: Add disfulency markers ('{f', '{c', etc) and '<laughter>' to the vocab. 
#       Also: consider resetting embedding weights for tokens that are in the 
#       vocab but have a specific annotation meaning ('[', /', --', '+', ...)
word_vectors = [[0] * glove_dim ]
idx2word = ['<UNK>'] 
with open('data/glove.6B/glove.6B.{}d.txt'.format(glove_dim)) as f:
    for i, line in enumerate(f.readlines()):
        line = line.split()
        idx2word.append(line[0])
        word_vectors.append([float(w) for w in line[1:]])
        if vocab_limit and i == vocab_limit:
            break
word2idx = {w: i for i,w in enumerate(idx2word)}
word_vectors = torch.FloatTensor(word_vectors)
embedding = nn.Embedding.from_pretrained(word_vectors)

# initialize the DAR RNN 
model = DARRNN(embedding, glove_dim, 100, nhidden, nlayers, dropout=0)

# load the training data
with open('data/swda_train.json') as f:
    train_data = json.load(f)
train_tags = [dialogue['tags'] for dialogue in train_data]
train_utts = [dialogue['utts'] for dialogue in train_data]

# TODO: Do we need additional pre-processing (clustering?) for tags?
#       see https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
tag_counter = Counter([t for ts in train_tags for t in ts])
idx2tag = [t for (t, count) in tag_counter.most_common()]
tag2idx = {t: i for i,t in enumerate(idx2tag)}

def utt2ids(utt):
    return [word2idx[w] if w in word2idx else word2idx['<UNK>'] for w in utt ]

model.train()
for utts, tags in zip(train_utts, train_tags):
    utts = [torch.LongTensor(utt2ids(utt)) for utt in utts]
    print(utts)
    break


