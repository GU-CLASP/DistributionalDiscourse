import torch
import torch.nn as nn
import torch.optim as optim

import model
import data

import itertools
import json 
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("encoder", choices=['wordvec-avg'], 
        help="Which encoder model to use")
parser.add_argument('--epochs', default=10,
        help='Number of times to iterate through the training data.')
parser.add_argument('--batch-size', type=int, default=10,
        help='Training batch size (number of dialogues).')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
        help='Training report interval (batches)')
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


def gen_batches(data, batch_size):
    x_batch, y_batch, i = [], [], 0
    for x, y in data:
        assert len(x) == len(y)
        x_batch.append(x)
        y_batch.append(y)
        i += 1
        if i == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch, i = [], [], 0
    if x_batch:
        yield x_batch, y_batch # final batch (possibly smaller than batch_size)

def train(utt_encoder_model, dar_model, train_data, n_epochs, batch_size, n_labels, 
        log_interval=200, train_encoder=True):

    if train_encoder:
        utt_encoder_model.train()
        train_params = itertools.chain(dar_model.parameters(), utt_encoder_model.parameters())
    else:
        train_params = dar_model.parameters()
    dar_model.train()
    optimizer = optim.Adam(train_params) 
    criterion = nn.CrossEntropyLoss()

    hidden = dar_model.init_hidden(batch_size)
    time_start = time.time()
    time_since_last_log = time_start
    prev_log_loss = 0
    for epoch in range(n_epochs):
        total_loss = 0
        batch_accuracy = []
        n_total_batches = int(len(train_data) / batch_size)
        for batch, (x, y) in enumerate(gen_batches(train_data, batch_size)):

            # zero out the gradients & detach history from the previous dialogue
            dar_model.zero_grad() 
            utt_encoder_model.zero_grad()
            hidden = dar_model.init_hidden(len(x)) # usually == batch_size except for possibly last batch.

            # Encode each utterance of each dialogue in the batch TODO: batchify this step
            x = [torch.stack([utt_encoder_model(torch.LongTensor(x_ij)) for x_ij in x_i]) for x_i in x]

            # Pad dialogues and DA labels to the max length (in utterances) for the batch 
            x = nn.utils.rnn.pad_sequence(x) 
            y = nn.utils.rnn.pad_sequence([torch.LongTensor(yi) for yi in y])

            # Make DA predictions 
            y_hat, hidden = dar_model(x, hidden)

            # Compute loss, backpropagate, and update model weights
            loss = criterion(y_hat.view(-1, n_labels), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_accuracy.append((y_hat.max(dim=2)[1] == y).sum().item() / y.numel())

            if batch % log_interval == 0 and batch > 0:
                cur_time = time.time()
                elapsed = cur_time - time_since_last_log
                print("Epoch {} | {:4d}/{:4d} batches | {:5.2f} ms/batch | Current loss: {:5.2f}" .format(
                    epoch, batch, n_total_batches, elapsed * 1000, total_loss - prev_log_loss))
                time_since_last_log = cur_time 
                prev_log_loss = total_loss

        print("Epoch {} | Total loss: {} | Mean batch accuracy: {}" .format(
            epoch, total_loss, sum(batch_accuracy) / len(batch_accuracy)))


if __name__ == '__main__':
    args = parser.parse_args()

    word_vocab, word2id = data.load_vocab(args.vocab_file)
    tag_vocab, tag2id = data.load_vocab(args.tag_vocab_file)

    with open(args.train_file) as f:
        train_data = json.load(f)
    train_tags = [item['tags_ints'] for item in train_data]
    train_utts = [item['utts_ints'] for item in train_data] 

    dar_model = model.DARRNN(args.utt_dims, len(tag_vocab), 200, 1, dropout=0)
    if args.encoder == 'wordvec-avg': 
        if args.use_glove:
            wordvectors = data.load_glove(args.utt_dims, word_vocab)
            utt_encoder_model = model.WordVecAvg.from_pretrained(torch.FloatTensor(wordvectors))
        else:
            utt_encoder_model = model.WordVecAvg.random_init(len(word_vocab), args.utt_dims)
    else:
        raise ValueError("Unknown encoder model: {}".format(args.encoder))

    # TODO load training data
    train(utt_encoder_model, dar_model, list(zip(train_utts, train_tags)), 
        args.epochs, args.batch_size, len(tag_vocab), log_interval=args.log_interval, 
        train_encoder=args.train_encoder)

