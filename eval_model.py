import data
import model
import util

import torch
import torch.nn as nn

import argparse
import json
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, 
        help='The save directory for the model to evaluate')
parser.add_argument('load_epoch', type=int, 
        help='The model checkpoint (training epoch) to load.')
parser.add_argument('--test-file', type=str, default='data/swda_test.json', 
        help='Path of the file containing test data.')
parser.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=logging.INFO, 
        help="Increase output verbosity")


def compute_accuracy(data, preds):
    tags = [diag[1] for diag in data]
    total, total_correct = 0, 0
    for y, y_hat in zip(tags, preds):
        len_y = len(y)
        assert(len_y == len(y_hat))
        total += len_y
        correct = sum([a == b for a,b in zip(y, y_hat)])
        total_correct += correct
    return total_correct / total

def eval_model(utt_encoder, dar_model, data, n_tags, criterion, device):
    """ Similar to train.train_epoch but:
        - runs through dialogues one at a time (instead of in batches)
        - hidden state retained for the whole dialogue (instead of using bptt sequences)
        - no backpropegation, obviously
        - returns predictions as well as loss
    """
    preds, loss = [], 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data, 1):
           log.debug("Evaluating on dialogue {:3d} of {} ({} utterances).".format(i, len(data), len(x)))
           x = [torch.LongTensor(xi).unsqueeze(0).to(device) for xi in x]
           y = torch.LongTensor(y).to(device)
           x = [utt_encoder(xi) for xi in x]
           x = torch.stack(x)
           hidden = dar_model.init_hidden(1)
           y_hat, hidden = dar_model(x, hidden)
           diag_loss = criterion(y_hat.view(-1, n_tags), y.view(-1)).item()
           log.debug("Dialogue {:3d} loss: {:.6f}".format(i, diag_loss))
           loss += diag_loss
           preds.append(y_hat.max(dim=2)[1].squeeze(1).tolist())
        loss = loss / len(data)
    return loss, preds 


if __name__ == '__main__':

    args = parser.parse_args()

    if not args.model_dir.startswith('models'):
        model_dir = os.path.join('models', args.model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'r') as f:
        model_args = json.load(f)
    log = util.create_logger(args.verbose, os.path.join(model_dir, 'eval.log'))
    args.__dict__.update(model_args)


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
        utt_encoder = model.BertUttEncoder(args.utt_dims) 
    else:
        raise ValueError("Unknown encoder model: {}".format(args.utt_encoder))

    # always use the same dar_model  
    dar_model = model.DARRNN(args.utt_dims, n_tags, args.dar_hidden, args.dar_layers, dropout=0)

    log.debug("Load the model state dicts.")
    utt_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'utt_encoder.E{}.bin'.format(args.load_epoch))))
    dar_model.load_state_dict(torch.load(os.path.join(model_dir, 'dar_model.E{}.bin'.format(args.load_epoch))))

    dar_model.eval()
    utt_encoder.eval()

    tag_format = 'tags_ints'
    test_data = data.load_data(args.test_file, utt_format, tag_format)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad targets don't contribute to the loss
    loss, preds = eval_model(utt_encoder, dar_model, test_data, n_tags, criterion, 'cpu')
    accuracy = compute_accuracy(test_data, preds)
    log.info("Test loss: {:.6f} | accuracy: %{:.2f}".format(loss, accuracy * 100))

    log.debug("Saving preds.json")
    with open(os.path.join(model_dir, 'preds.E{}.json'.format(args.load_epoch)), 'w') as f:
        json.dump(preds, f)

