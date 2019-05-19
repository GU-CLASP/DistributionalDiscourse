import torch
import torch.nn as nn

import math 
import contextlib
import logging
from tqdm import tqdm

log = logging.getLogger('DARLOGGER')

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

def run_model(mode, utt_encoder, dar_model, data, n_tags, utt_batch_size, diag_batch_size, 
    optimizer=None, device=torch.device('cpu')):

    assert mode in ('train', 'evaluate')

    total_batches = math.ceil(len(data) / diag_batch_size)
    batches = gen_diag_batches(data, diag_batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad targets don't contribute to the loss
    batch_losses, preds = [], []

    # disable gradients unless we're training
    if mode == 'train':
        if not optimizer:
            raise ValueError("Must supply an optimizer in train mode.")
        mode_context = contextlib.nullcontext()
    else:
        mode_context = torch.no_grad()
    
    with mode_context:  
        for batch, (x, y, diag_batch_size_) in tqdm(enumerate(batches), total=total_batches):

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
                    utt_lens = torch.FloatTensor([max(len(utt), 25) for utt in utts]).to(device)
                    utts = [torch.LongTensor(utt[-25:]) for utt in utts]
                    # Pad the utterances in the dialogue to the max length utt length in the dialogue
                    utts = nn.utils.rnn.pad_sequence(utts, batch_first=True).to(device)
                    utts = utt_encoder(utts, utt_lens)
                    encoded_diag.append(utts)
                encoded_diag = torch.cat(encoded_diag)
                encoded_diags.append(encoded_diag)
            x = encoded_diags 

            # Pad dialogues and DA tags to the max length (in utterances) for the batch 
            diag_lens = [len(xi) for xi in x]
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
            batch_losses.append(batch_loss)
            batch_preds = y_hat.max(dim=2)[1]  
            batch_preds = [batch_preds[:,i][:diag_lens[i]].tolist() for i in range(diag_batch_size_)]
            preds += batch_preds

            log.debug('Batch {} loss {:.2f} ({})'.format(batch+1, batch_loss, mode))

    average_loss = sum(batch_losses) / total_batches
    return average_loss, preds
