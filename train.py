import torch
import torch.nn as nn
import torch.optim as optim

import model

import json 
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("encoder", choices=['wordvec-avg'], 
        help="Which encoder model to use")
parser.add_argument('--epochs', default=10,
        help='Number of times to iterate through the training data.')
parser.add_argument('--batch-size', default=10,
        help='Training batch size (number of dialogues).')
parser.add_argument('--wordvec-file', type=str, default=None, 
        help='Path of a Glove format word vector file.')
parser.add_argument('--vocab-file', type=str, default='data/vocab.json', 
        help='Path of the vocabulary to use (list of words).')
parser.add_argument('--train-encoder', action='store_true', default=False,
        help='Train the utterance encoder. (Otherwise only the DAG RNN is trained.)')
parser.add_argument('--utt-dims', default=100,
        help='Set the number of dimensions of the utterance embedding.'
        'For wordvec-* models, this is equal to the word vector size.')


def gen_batches(data, batch_size):
    x_batch, y_batch, i = [], [], 0
    for x, y in data:
        x_batch.append(x)
        y_batch.append(y)
        i += 1
        if i == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch, i = [], [], 0
    if x_batch:
        yield x_batch, y_batch # final batch (possibly smaller than batch_size)

def train(utt_encoder_model, dar_model, train_data, n_epochs, batch_size, n_labels, train_encoder=True):

    if train_encoder:
        utt_encoder_model.train()
        train_params = itertools.chain(dar_model.parameters(), utt_encoder_model.parameters())
    else:
        train_params = dar_model.parameters(), utt_encoder_model.parameters()
    dar_model.train()
    optimizer = optim.Adam(train_params) 
    criterion = nn.CrossEntropyLoss()

    hidden = dar_model.init_hidden(batch_size)
    for epoch in range(n_epochs):
        total_loss = 0
        batch_accuracy = []
        for x, y in gen_batches(train_data, batch_size):

            # zero out the gradients & detach history from the previous dialogue
            dar_model.zero_grad() 
            utt_encoder_model.zero_grad()
            hidden = dar_model.init_hidden(batch_size) 

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

        print("Epoch {} | Total loss: {} | Mean batch accuracy: {}" .format(
            epoch, total_loss, sum(batch_accuracy) / len(batch_accuracy)))


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.vocab_file) as f:
        vocab_word2id = json.load(f) # list of tokens 
    vocab_id2word = [

    dar_model = model.DARRNN(args.utt_dims, len(label_vocab), 200, 1, dropout=0)
    if args.encoder == 'wordvec-avg': 
        with open('data/glove.6B/glove.6B.{}d.txt'.format(args.utt_dims)) as f:
            wordvectors = {line[0]: list(map(float, line[1:])) for line in f.readlines()}
        wordvectors = [wordvectors[vocab[w]] for w in vocab]  # order the word vectors according to the vocab
        utt_encoder_model = model.Word2VecAvg.from_pretrained(torch.FloatTensor(wordvectors))
    else:
        raise ValueError("Unknown encoder model: {}".format(args.encoder))

    # TODO load training data
    train(utt_encoder_model, dar_model, train_data, 
        args.epochs, args.batch_size, len(label_vocab), train_encoder=args.train_encoder)

