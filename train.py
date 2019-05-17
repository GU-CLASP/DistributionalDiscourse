import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer

import model
import data

from tqdm import tqdm
import contextlib
import itertools
import json 
import argparse
import math 

parser = argparse.ArgumentParser()
parser.add_argument("utt_encoder", choices=['wordvec-avg', 'bert'], 
        help="Which utt_encoder model to use")
parser.add_argument('--epochs', default=10,
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
parser.add_argument('--freeze-encoder', action='store_true', default=False,
        help='Train the utterance encoder. (Otherwise only the DAG RNN is trained.)')
parser.add_argument('--glove', dest='use_glove', action='store_true')
parser.add_argument('--no-glove', dest='use_glove', action='store_false')
parser.set_defaults(use_glove=True)
parser.add_argument('--utt-dims', default=100, type=int,
        help='Set the number of dimensions of the utterance embedding.'
        'For wordvec-* models, this is equal to the word vector size.')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--train-log-interval', default=None,
        help='Report on training accuracy/loss every N batches.')

def gen_diag_batches(data, batch_size):
    """ Yields batches of batch_size from data, where data is a list of swda dialogues
        swda_tokenizer should take a dialogue and return tokenized utterances [[int]]
            and tokenized act labels [int] 
    """
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

def gen_utt_batches(data, batch_size):
    """ Essentially the same as gen_diag_batches, but with no labels
    """
    x_batch, i = [], 0
    for x in data:
        x_batch.append(x)
        i += 1
        if i == batch_size:
            yield x_batch, batch_size
            x_batch, i = [], 0
    if x_batch:
        yield x_batch, len(x_batch)  # final batch (possibly smaller than batch_size)

def run_model(mode, utt_encoder, dar_model, train_data, n_tags, 
    utt_batch_size, diag_batch_size, epoch, device=torch.device('cpu'), print_every=None):

    assert mode in ('train', 'evaluate')

    total_loss, total_correct, total_items = 0, 0, 0
    total_batches = math.ceil(len(train_data) / diag_batch_size)
    batches = gen_diag_batches(train_data, diag_batch_size)

    # disable gradients unless we're training
    if mode == 'train':
        mode_context = contextlib.nullcontext()
    else:
        mode_context = torch.no_grad()
    
    with mode_context:  
        for batch, (x, y, diag_batch_size_) in tqdm(enumerate(batches), 
                desc='Epoch {}'.format(epoch), total=total_batches):

            if mode == 'train':
            # zero out the gradients & detach history from the previous batch of dialogues
                dar_model.zero_grad() 
                utt_encoder.zero_grad()
            hidden = dar_model.init_hidden(diag_batch_size_) 

            encoded_diags = []
            for diag in tqdm(x, total=diag_batch_size_, desc="Encoding batch {}".format(batch)):
                encoded_diag = []
                utt_batches = gen_utt_batches(diag, utt_batch_size)
                for utts, utt_batch_size_, in utt_batches:
                    # Keep track of the real lengths so we can mask inputs
                    utt_lens = torch.FloatTensor([len(utt) for utt in utts])
                    utts = [torch.LongTensor(utt) for utt in utts]
                    # Pad the utterances in the dialogue to the max length utt length in the dialogue
                    utts = nn.utils.rnn.pad_sequence(utts, batch_first=True)
                    utts = utt_encoder(utts, utt_lens)
                    encoded_diag.append(utts)
                encoded_diag = torch.cat(encoded_diag)
                encoded_diags.append(encoded_diag)
            x = encoded_diags 

            # Pad dialogues and DA tags to the max length (in utterances) for the batch 
            x = nn.utils.rnn.pad_sequence(x).to(device)
            y = nn.utils.rnn.pad_sequence([torch.LongTensor(yi) for yi in y]).to(device)

            # Make DA predictions 
            y_hat, hidden = dar_model(x, hidden)

            # Compute loss, backpropagate, and update model weights
            loss = criterion(y_hat.view(-1, n_tags), y.view(-1))
            if mode == 'train':
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            batch_correct = (y_hat.max(dim=2)[1] == y).sum().item() 
            total_loss += batch_loss 
            total_correct += batch_correct
            total_items += y.numel()
          
            if print_every and batch % print_every == 0:
                tqdm.write('Epoch {} {} batch {} | batch loss {:.2f} | accuracy {:.2f}'.format(
                    epoch+1, mode, batch+1, batch_loss, batch_correct / y.numel()))

    tqdm.write('Epoch {} {} | total loss {:.2f} | accuracy {:.2f}'.format(
        epoch+1, mode, total_loss, total_correct / total_items))

    accuracy = total_correct / total_items 
    return utt_encoder, dar_model, (total_loss, accuracy)

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print("Training on {}.".format(device))

    word_vocab, word2id = data.load_vocab(args.vocab_file)
    tag_vocab, tag2id = data.load_vocab(args.tag_vocab_file)
    n_tags = len(tag_vocab)

    # select an utt_encoder and compatible utt tokenization
    print("Utt encoder: {}".format(args.utt_encoder))
    if args.utt_encoder == 'wordvec-avg': 
        if args.use_glove:
            embedding = torch.FloatTensor(data.load_glove(args.utt_dims, word_vocab))
            embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        else:
            embedding = nn.Embedding(len(word_vocab), args.utt_dims, padding_idx=0)
        utt_format = 'utts_ints'
        utt_dims = args.utt_dims
        utt_encoder = model.WordVecAvg(embedding)
    elif args.utt_encoder == 'bert':
        utt_format = 'utts_ints_bert'
        utt_dims = 768  # ignore args.utt_dims for BERT
        utt_encoder = model.BertUttEncoder.from_pretrained_base_uncased()
    else:
        raise ValueError("Unknown encoder model: {}".format(args.utt_encoder))
    utt_encoder.to(device)

    # always use the same dar_model
    dar_model = model.DARRNN(utt_dims, n_tags, 200, 1, dropout=0)
    dar_model.to(device)

    # select the parameters to train
    if args.freeze_encoder: 
        train_params = dar_model.parameters()
    else:
        utt_encoder.train()
        train_params = itertools.chain(dar_model.parameters(), utt_encoder.parameters())

    tag_format = 'tags_ints'
    train_data = data.load_data(args.train_file, utt_format, tag_format)
    val_data = data.load_data(args.val_file, utt_format, tag_format)

    optimizer = optim.Adam(train_params) 
    criterion = nn.CrossEntropyLoss()

    dar_model.train()
    for epoch in range(args.epochs):
        utt_encoder, dar_model, _ = run_model('train', utt_encoder, dar_model, train_data, n_tags, 
                                        args.utt_batch_size, args.diag_batch_size, epoch, device, 
                                        print_every=args.train_log_interval)
        utt_encoder, dar_model, _= run_model('evaluate', utt_encoder, dar_model, val_data, n_tags,
                                        args.utt_batch_size, 1, epoch, device)
