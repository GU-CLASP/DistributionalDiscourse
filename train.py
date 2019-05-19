import model
import data
import util
from run_model import run_model

import torch
import torch.nn as nn
import torch.optim as optim

import itertools
import json 
import argparse
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument("utt_encoder", choices=['wordvec-avg', 'bert'], 
        help="Which utt_encoder model to use")
parser.add_argument('--utt-dims', default=100, type=int,
        help='Set the number of dimensions of the utterance embedding.'
        'For wordvec-* models, this is equal to the word vector size.')
parser.add_argument('--dar-hidden', default=100, type=int,
        help="Size of the hidden layer in the DAR RNN.")
parser.add_argument('--dar-layers', default=1, type=int,
        help="Number of hidden layers in the DAR RNN.")
parser.add_argument('--freeze-encoder', action='store_true', default=False,
        help='Train the utterance encoder. (Otherwise only the DAG RNN is trained.)')
parser.add_argument('--glove', dest='use_glove', action='store_true', default=False,
        help="Use GloVe (with compatible utt encoders).")
parser.add_argument('--epochs', type=int, default=10,
        help='Number of times to iterate through the training data.')
parser.add_argument('--diag-batch-size', type=int, default=10,
        help='Size of dialogue batches (for DAR seq2seq)')
parser.add_argument('--utt-batch-size', type=int, default=10,
        help='Size of utterance batches (for utt encoding)')
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
parser.add_argument('--save-suffix', type=str, default='', 
        help='A suffix to add to the name of the save directory.')
parser.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=logging.INFO, 
        help="Increase output verbosity")
        
    

if __name__ == '__main__':

    args = parser.parse_args()
    save_dir = os.path.join('models', args.utt_encoder + args.save_suffix) 

    # create the save directory (for trianed model paremeters, logs, arguments)
    if not os.path.exists('models'):
        os.mkdir('models')
    if os.path.exists(save_dir):
        go_ahead = input("Overwriting files in {}. Continue? (y/n): ".format(save_dir))
        util.rm_dir(save_dir)
        if not go_ahead == 'y':
            exit()
    os.mkdir(save_dir)
   
    # save the args so we can recover hyperparameters, etc.
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    log = util.create_logger(args.verbose, os.path.join(save_dir, 'train.log'))

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    log.info("Training on {}.".format(device))

    word_vocab, word2id = data.load_vocab(args.vocab_file)
    tag_vocab, tag2id = data.load_vocab(args.tag_vocab_file)
    n_tags = len(tag_vocab)

    # select an utt_encoder and compatible utt tokenization
    log.info("Utt encoder: {}".format(args.utt_encoder))
    if args.utt_encoder == 'wordvec-avg': 
        if args.use_glove:
            weights = torch.FloatTensor(data.load_glove(args.utt_dims, word_vocab))
            utt_encoder = model.WordVecAvg.from_pretrained(weights)
        else:
            utt_encoder = model.WordVecAvg.random_init(len(word_vocab), args.utt_dims)
        utt_format = 'utts_ints'
        utt_dims = args.utt_dims
    elif args.utt_encoder == 'bert':
        utt_format = 'utts_ints_bert'
        utt_dims = 768  # ignore args.utt_dims for BERT
        utt_encoder = model.BertUttEncoder.from_pretrained_base_uncased()
    else:
        raise ValueError("Unknown encoder model: {}".format(args.utt_encoder))

    # always use the same dar_model
    dar_model = model.DARRNN(utt_dims, n_tags, args.dar_hidden, args.dar_layers, dropout=0)

    # select the parameters to train
    if args.freeze_encoder: 
        train_params = dar_model.parameters()
    else:
        utt_encoder.train()
        train_params = itertools.chain(dar_model.parameters(), utt_encoder.parameters())
    dar_model.train()

    tag_format = 'tags_ints'
    train_data = data.load_data(args.train_file, utt_format, tag_format)
    val_data = data.load_data(args.val_file, utt_format, tag_format)

    optimizer = optim.Adam(train_params) 

    dar_model.to(device)
    utt_encoder.to(device)

    for epoch in range(args.epochs):
        run_model('train', utt_encoder, dar_model, train_data, n_tags,
                args.utt_batch_size, args.diag_batch_size, epoch,
                optimizer=optimizer, device=device)
        run_model('evaluate', utt_encoder, dar_model, val_data, n_tags,
                args.utt_batch_size, 1, epoch, device=device)
        
    torch.save(dar_model.state_dict(), os.path.join(save_dir, 'dar_model.bin'))
    torch.save(utt_encoder.state_dict(), os.path.join(save_dir, 'utt_encoder.bin'))
