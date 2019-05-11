import torch
import torch.nn as nn
import torch.optim as optim

import model
import data

from tqdm import tqdm
import itertools
import json 
import argparse
import math 

parser = argparse.ArgumentParser()
parser.add_argument("utt_encoder", choices=['wordvec-avg'], 
        help="Which utt_encoder model to use")
parser.add_argument('--epochs', default=10,
        help='Number of times to iterate through the training data.')
parser.add_argument('--batch-size', type=int, default=10,
        help='Training batch size (number of dialogues).')
parser.add_argument('--vocab-file', type=str, default='data/swda_vocab.json', 
        help='Path of the vocabulary to use (id -> word dict).')
parser.add_argument('--tag-vocab-file', type=str, default='data/swda_tag_vocab.json', 
        help='Path of the tag vocabulary to use (id -> tag dict).')
parser.add_argument('--train-file', type=str, default='data/swda_train.json', 
        help='Path of the file containing training data.')
parser.add_argument('--train-encoder', action='store_true', default=False,
        help='Train the utterance encoder. (Otherwise only the DAG RNN is trained.)')
parser.add_argument('--glove', dest='use_glove', action='store_true')
parser.add_argument('--no-glove', dest='use_glove', action='store_false')
parser.set_defaults(use_glove=True)
parser.add_argument('--utt-dims', default=100, type=int,
        help='Set the number of dimensions of the utterance embedding.'
        'For wordvec-* models, this is equal to the word vector size.')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--verbose', action='store_true', default=False,
        help='How much to print during training')


def gen_batches(data, batch_size):
    x_batch, y_batch, i = [], [], 0
    for x, y in data:
        x_batch.append(x)
        y_batch.append(y)
        i += 1
        if i == batch_size:
            yield x_batch, y_batch, batch_size
            x_batch, y_batch, i = [], [], 0
    if x_batch:
        yield x_batch, y_batch, len(x_batch)  # final batch (possibly smaller than batch_size)

def train(utt_encoder, dar_model, train_data, n_epochs, batch_size, n_tags, train_encoder=True, device=torch.device('cpu'), verbose=False):

    if train_encoder: 
        utt_encoder.train()
        train_params = itertools.chain(dar_model.parameters(), utt_encoder.parameters())
    else:
        train_params = dar_model.parameters()
    dar_model.train()
    optimizer = optim.Adam(train_params) 
    criterion = nn.CrossEntropyLoss()

    hidden = dar_model.init_hidden(batch_size)
    for epoch in range(n_epochs):
        total_loss = 0
        batch_accuracy = []
        total_batches = math.ceil(len(train_data) / batch_size)
        for batch, (x, y, actual_batch_size) in tqdm(enumerate(gen_batches(train_data, batch_size)), 
                desc='Epoch {}'.format(epoch), total=total_batches):

            # zero out the gradients & detach history from the previous dialogue
            dar_model.zero_grad() 
            utt_encoder.zero_grad()
            hidden = dar_model.init_hidden(actual_batch_size) 

            # Encode each utterance of each dialogue in the batch TODO: batchify this step
            x = [torch.stack([utt_encoder(torch.LongTensor(x_ij)) for x_ij in x_i]) for x_i in x]

            # Pad dialogues and DA tags to the max length (in utterances) for the batch 
            x = nn.utils.rnn.pad_sequence(x).to(device)
            y = nn.utils.rnn.pad_sequence([torch.LongTensor(yi) for yi in y]).to(device)

            # Make DA predictions 
            y_hat, hidden = dar_model(x, hidden)

            # Compute loss, backpropagate, and update model weights
            loss = criterion(y_hat.view(-1, n_tags), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_accuracy.append((y_hat.max(dim=2)[1] == y).sum().item() / y.numel())
            
            if verbose:
                tqdm.write('Batch {} loss: {:.2f} '.format(batch+1, loss.item()))

        print("Total loss: {:.2f} | Mean batch accuracy: {:.4f}" .format(
            total_loss, sum(batch_accuracy) / total_batches))


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print("Training on {}.".format(device))

    word_vocab, word2id = data.load_vocab(args.vocab_file)
    tag_vocab, tag2id = data.load_vocab(args.tag_vocab_file)

    with open(args.train_file) as f:
        train_data = json.load(f)
    train_tags = [item['tags_ints'] for item in train_data]
    train_utts = [item['utts_ints'] for item in train_data] 

    dar_model = model.DARRNN(args.utt_dims, len(tag_vocab), 200, 1, dropout=0)
    if args.utt_encoder == 'wordvec-avg': 
        if args.use_glove:
            embedding = torch.FloatTensor(data.load_glove(args.utt_dims, word_vocab))
            embedding = nn.Embedding.from_pretrained(embedding)
        else:
            embedding = nn.Embedding(len(word_vocab), args.utt_dims)
        utt_encoder = model.WordVecAvg(embedding)
    else:
        raise ValueError("Unknown encoder model: {}".format(args.utt_encoder))

    dar_model.to(device)
    utt_encoder.to(device)
    train(utt_encoder, dar_model, list(zip(train_utts, train_tags)), 
        args.epochs, args.batch_size, len(tag_vocab), train_encoder=args.train_encoder,
        device=device, verbose=args.verbose)

