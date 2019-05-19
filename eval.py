import data
import model
from run_model import run_model

import torch
import torch.nn as nn

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, 
        help='The save directory for the model to evaluate')
parser.add_argument('--test-file', type=str, default='data/swda_test.json', 
        help='Path of the file containing test data.')

if __name__ == '__main__':

    args = parser.parse_args()

    if not args.model_dir.startswith('models'):
        model_dir = os.path.join('models', args.model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        model_args = json.load(f)
    args.__dict__.update(model_args)

    log = util.create_logger(args.verbose, os.path.join(save_dir, 'eval.log'))

    word_vocab, word2id = data.load_vocab(args.vocab_file)
    tag_vocab, tag2id = data.load_vocab(args.tag_vocab_file)
    n_tags = len(tag_vocab)

    # select an utt_encoder and compatible utt tokenization
    log.info("Utt encoder: {}".format(args.utt_encoder))
    if args.utt_encoder == 'wordvec-avg': 
        utt_format = 'utts_ints'
        embedding = nn.Embedding(len(word_vocab), args.utt_dims, padding_idx=0)
        utt_encoder = model.WordVecAvg(embedding)
    elif args.utt_encoder == 'bert':
        utt_format = 'utts_ints_bert'
        # TODO: maybe use BertModel() with an explicit config to avoid loading weights twice
        utt_encoder = model.BertUttEncoder.from_pretrained_base_uncased() 
    else:
        raise ValueError("Unknown encoder model: {}".format(args.utt_encoder))

    # always use the same dar_model  
    dar_model = model.DARRNN(args.utt_dims, n_tags, args.dar_hidden, args.dar_layers, dropout=0)

    # load the model state dicts
    utt_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'utt_encoder.bin')))
    dar_model.load_state_dict(torch.load(os.path.join(model_dir, 'dar_model.bin')))

    dar_model.eval()
    utt_encoder.eval()

    tag_format = 'tags_ints'
    test_data = data.load_data(args.test_file, utt_format, tag_format)

    criterion = nn.CrossEntropyLoss()

    run_model('evaluate', utt_encoder, dar_model, test_data, n_tags, criterion, None,
            args.utt_batch_size, 1, 0, torch.device('cpu'))
