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
parser.add_argument('corpus', choices=['SWDA', 'AMI-DA'],
        help='Which dialouge act corpus to eval on.')
parser.add_argument('-e', '--epoch', type=int, default=None
        help='The model checkpoint (training epoch) to load.')
parser.add_argument('-d','--data-dir', default='data',
        help='Data storage directory.')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--gpu-id', type=int, default=0,
        help='Select with GPU to use')
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

def get_max_val_loss(model_dir):
    """
    model_dir - the full path to the save directory for the model
    returns: a tuple containing the training epoch (stating with 1),
             with the maxiumum validation accuracy, and its value 
    """
    val_loss_re = re.compile(r"\[INFO    \] Epoch (\d) validation loss: \d+\.\d+ \| accuracy: %(\d+\.\d+)")
    with open(os.path.join(model_dir, 'train.log')) as f:
        train_log = f.read()
    epoch_val_loss = re.findall(val_loss_re, train_log)
    max_epoch, max_loss = max(epoch_val_loss, key=lambda x:x[1])
    return max_epoch, max_loss

def eval_model(encoder_model, dar_model, data, n_tags, criterion, device, min_utt_len=None):
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
           x = util.pad_lists(x, min_len=min_utt_len)
           x = [torch.LongTensor(xi).unsqueeze(0).to(device) for xi in x]
           x = [encoder_model(xi) for xi in x]
           x = torch.stack(x)
           y = torch.LongTensor(y).to(device)
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
    if not args.epoch:
        args.epoch = get_max_val_loss(args.model_dir)
    with open(os.path.join(args.model_dir, 'args.json'), 'r') as f:
        model_args = json.load(f)
    args.__dict__ = dict(list(model_args.items()) + list(args.__dict__.items()))
    log = util.create_logger(args.verbose, os.path.join(args.model_dir, 'eval.log'))

    device = torch.device(f'cuda:{args.gpu_id}' if args.cuda and torch.cuda.is_available() else 'cpu')
    log.info(f"Evaluating on {device}.")

    tag_vocab_file = os.path.join(args.data_dir, f'{args.corpus}_tags.txt')
    test_file = os.path.join(args.data_dir, f'{args.corpus}_test.json')
    preds_file = os.path.join(args.model_dir, f'preds.E{args.epoch}.json')
    encoder_model_file = os.path.join(args.model_dir, f'encoder_model.E{args.epoch}.bin')
    dar_model_file = os.path.join(args.model_dir, f'dar_model.E{args.epoch}.bin')

    tokenizer = data.load_tokenizer('bert-base-uncased')
    vocab_size = len(tokenizer)
    tag_vocab, tag2id = data.load_tag_vocab(tag_vocab_file)
    n_tags = len(tag_vocab)

    # select an encoder_model and compatible utt tokenization
    log.info("Utt encoder: {}".format(args.encoder_model))
    if args.encoder_model == 'wordvec-avg': 
        encoder_model = model.WordVecAvg.random_init(vocab_size, args.embedding_size)
    elif args.encoder_model == 'cnn':
        window_sizes = [3, 4, 5]
        feature_maps = 100
        min_utt_len = max(window_sizes) 
        encoder_model = model.KimCNN.random_init(vocab_size, args.utt_dims, 
                args.embedding_size, window_sizes, feature_maps)
    elif args.encoder_model == 'bert':
        encoder_model = model.BertEncoder(args.utt_dims,
                from_pretrained=False, finetune_bert=False,
                resize=len(tokenizer)) 
    else:
        raise ValueError("Unknown encoder model: {}".format(args.encoder_model))

    # always use the same dar_model  
    dar_model = model.DARRNN(args.utt_dims, n_tags, args.dar_hidden, args.dar_layers, dropout=0)

    log.debug("Load the model state dicts.")
    encoder_model.load_state_dict(torch.load(encoder_model_file))
    dar_model.load_state_dict(torch.load(dar_model_file))

    dar_model.eval()
    dar_model.to(device)
    encoder_model.eval()
    encoder_model.to(device)

    test_data = data.load_data(test_file, tokenizer, tag2id, strip_laughter=args.no_laughter)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad targets don't contribute to the loss
    loss, preds = eval_model(encoder_model, dar_model, test_data, n_tags, criterion, device)
    accuracy = compute_accuracy(test_data, preds)
    log.info(f"Test loss: {loss:.6f} | accuracy: %{accuracy*100:.2f}")

    preds = [[tag_vocab[t] for t in dialogue] for dialogue in preds]
    log.debug(f"Saving {preds_file}")
    with open(preds_file, 'w') as f:
        json.dump(preds, f)

