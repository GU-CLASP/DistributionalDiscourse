import model
import data
import util
import eval_model

import torch
import torch.nn as nn
import torch.optim as optim

import itertools
import json 
import argparse
import os
import logging
import random
import math 
import contextlib
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("utt_encoder", choices=['wordvec-avg', 'bert'], 
        help="Which utt_encoder model to use")
parser.add_argument('--utt-dims', default=100, type=int,
        help='Set the number of dimensions of the utterance embedding.'
        'For wordvec-* models, this is equal to the word vector size.')
parser.add_argument('--dar-hidden', default=100, type=int,
        help="Size of the hidden layer in the DAR RNN.")
parser.add_argument('--lstm', action='store_true',
        help="Use an LSTM for the DAR RNN.")
parser.add_argument('--dar-layers', default=1, type=int,
        help="Number of hidden layers in the DAR RNN.")
parser.add_argument('--freeze-encoder', action='store_true', default=False,
        help='Train the utterance encoder. (Otherwise only the DAG RNN is trained.)')
parser.add_argument('--glove', dest='use_glove', action='store_true', default=False,
        help="Use GloVe (with compatible utt encoders).")
parser.add_argument('--epochs', type=int, default=10,
        help='Number of times to iterate through the training data.')
parser.add_argument("--learning-rate", default=3e-5, type=float,
        help="The initial learning rate for Adam.")
parser.add_argument('--batch-size', type=int, default=10,
        help='Size of dialogue batches (for DAR seq2seq)')
parser.add_argument('--bptt', type=int, default=5,
        help='Length of sequences for backpropegation through time')
parser.add_argument('--max-utt-len', type=int, default=50,
        help='Maximum utterance length (truncates first part of long utterances).')
parser.add_argument('--vocab-file', type=str, default='data/swda_vocab.json', 
        help='Path of the vocabulary to use (id -> word dict).')
parser.add_argument('--tag-vocab-file', type=str, default='data/swda_tag_vocab.json', 
        help='Path of the tag vocabulary to use (id -> tag dict).')
parser.add_argument('--train-file', type=str, default='data/swda_train.json', 
        help='Path of the file containing training data.')
parser.add_argument('--val-file', type=str, default='data/swda_val.json', 
        help='Path of the file containing validation data.')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--gpu-id', type=int, default=0,
        help='Select with GPU to use')
parser.add_argument('--save-suffix', type=str, default='', 
        help='A suffix to add to the name of the save directory.')
parser.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=logging.INFO, 
        help="Increase output verbosity")
parser.add_argument("--training-limit", type=int, default=None,
        help="Limit the amount of training data to N dialogues.")

def pad_lists(ls, max_len=None, pad=0):
    pad_len = max(len(l) for l in ls)
    if max_len:
        pad_len = min(pad_len, max_len)
    return [(l + ([pad] * (pad_len - len(l))))[-pad_len:] for l in ls]

def gen_batches(data, batch_size):
    data.sort(key=lambda x: len(x[0]))  # batch similarly lengthed dialogues together
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    random.shuffle(batches)  # shuffle the batches so we a mix of lengths
    return batches

def gen_bptt(batch, bptt, batch_size, max_utt_len):
    utts_batch, tags_batch = zip(*batch)
    diag_lens = [len(tags) for tags in tags_batch]
    max_diag_len = max(diag_lens)
    utts_batch = [[utts_batch[i][j] if j < diag_lens[i] else []
            for i in range(batch_size)] for j in range(max_diag_len)]
    utts_batch = [pad_lists(utts, max_utt_len) for utts in utts_batch]
    tags_batch = [[tags_batch[i][j] if j < diag_lens[i] else 0
            for i in range(batch_size)] for j in range(max_diag_len)]
    for seq in range(0, max_diag_len, bptt):
        yield utts_batch[seq:seq+bptt], tags_batch[seq:seq+bptt]

def train_epoch(utt_encoder, dar_model, data, n_tags, batch_size, bptt, max_utt_len,
        criterion, optimizer, device):
    epoch_loss = 0
    batches = gen_batches(data, batch_size)
    for i, batch in enumerate(tqdm(batches), 1):
        batch_loss = 0
        batch_size_ = len(batch)
        hidden = dar_model.init_hidden(batch_size_) 
        for x, y in gen_bptt(batch, bptt, batch_size_, max_utt_len):
            # detach history from the previous batch
            hidden = hidden.detach() 
            # zero out the gradients 
            dar_model.zero_grad() 
            utt_encoder.zero_grad()
            # create tensors
            y = torch.LongTensor(y).to(device)
            # encode utterances (once for each item in the BPTT sequence)
            x = [torch.LongTensor(xi).to(device) for xi in x]
            x = [utt_encoder(xi) for xi in x]
            x = torch.stack(x)
            # predict DA tag sequences
            y_hat, hidden = dar_model(x, hidden) 
            # compute loss, backpropagate, and update model weights
            loss = criterion(y_hat.view(-1, n_tags), y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss += loss.item()
        batch_loss = batch_loss / batch_size_
        epoch_loss += batch_loss 
        log.debug('Batch {} loss {:.6f}'.format(i, batch_loss))
    epoch_loss = epoch_loss / i
    return epoch_loss

if __name__ == '__main__':

    args = parser.parse_args()
    save_dir = os.path.join('models', args.utt_encoder + args.save_suffix) 

    # create the save directory (for trianed model paremeters, logs, arguments)
    if not os.path.exists('models'):
        os.mkdir('models')
    if os.path.exists(save_dir):
        go_ahead = input("Overwriting files in {}. Continue? (y/n): ".format(save_dir))
        if go_ahead == 'y':
            util.rm_dir(save_dir)
        else:
            exit()
    os.mkdir(save_dir)
   
    # save the args so we can recover hyperparameters, etc.
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    log = util.create_logger(args.verbose, os.path.join(save_dir, 'train.log'))
    eval_model.log = log  # set the eval_model logger to go to 'train.log'

    device = torch.device('cuda:{}'.format(args.gpu_id) if args.cuda and torch.cuda.is_available() else 'cpu')
    log.info("Training on {}.".format(device))

    word_vocab, word2id = data.load_vocab(args.vocab_file)
    tag_vocab, tag2id = data.load_vocab(args.tag_vocab_file)
    n_tags = len(tag_vocab)

    # select an utt_encoder and compatible utt tokenization
    log.info("Utt encoder: {}".format(args.utt_encoder))
    log.info("DAR model uses LSTM: {}".format(args.lstm))
    if args.utt_encoder == 'wordvec-avg': 
        if args.use_glove:
            weights = torch.FloatTensor(data.load_glove(args.utt_dims, word_vocab))
            utt_encoder = model.WordVecAvg.from_pretrained(weights)
        else:
            utt_encoder = model.WordVecAvg.random_init(len(word_vocab), args.utt_dims)
        utt_format = 'utts_ints'
    elif args.utt_encoder == 'bert':
        utt_format = 'utts_ints_bert'
        utt_encoder = model.BertUttEncoder(args.utt_dims)
    else:
        raise ValueError("Unknown encoder model: {}".format(args.utt_encoder))

    # always use the same dar_model
    dar_model = model.DARRNN(args.utt_dims, n_tags, args.dar_hidden, args.dar_layers, dropout=0, use_lstm=args.lstm)
    # dar_model = model.SimpleDARRNN(args.utt_dims, n_tags)

    # select the parameters to train
    if args.freeze_encoder: 
        for p in utt_encoder.parameters():
            p.requires_grad = False
    else:
        utt_encoder.train()
    dar_model.train()

    params = list(dar_model.named_parameters()) + list(utt_encoder.named_parameters())
    log.debug("Model parameters ({} total):".format(len(params)))
    for n, p in params:
        log.debug("{:<25} | {:<10} | {}".format(
            str(p.size()),
            'training' if p.requires_grad else 'frozen',
            n if n else '<unnamed>'))

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad targets don't contribute to the loss
    optimizer = optim.Adam([p for n, p in params], lr=args.learning_rate)

    dar_model.to(device)
    utt_encoder.to(device)

    tag_format = 'tags_ints'
    train_data = data.load_data(args.train_file, utt_format, tag_format)
    val_data = data.load_data(args.val_file, utt_format, tag_format)
    if args.training_limit:
        train_data = train_data[:args.train_limit]
        val_data = val_data[:int(args.train_limit/2)]

    for epoch in range(1, args.epochs+1):
        log.info("Starting epoch {}".format(epoch))
        train_loss = train_epoch(utt_encoder, dar_model, train_data, n_tags,
                args.batch_size, args.bptt, args.max_utt_len, 
                criterion, optimizer, device)
        log.info("Epoch {} training loss:   {:.6f}".format(epoch, train_loss))
        log.info("Saving epoch {} models.".format(epoch))
        torch.save(dar_model.state_dict(), os.path.join(save_dir, 'dar_model.E{}.bin'.format(epoch)))
        torch.save(utt_encoder.state_dict(), os.path.join(save_dir, 'utt_encoder.E{}.bin'.format(epoch)))
        log.info("Starting epoch {} valdation".format(epoch))
        val_loss, preds = eval_model.eval_model(utt_encoder, dar_model, val_data, n_tags, 
                criterion, device)
        accuracy = eval_model.compute_accuracy(val_data, preds)
        log.info("Epoch {} validation loss: {:.6f} | accuracy: %{:.2f}".format(
            epoch, val_loss, accuracy * 100))
