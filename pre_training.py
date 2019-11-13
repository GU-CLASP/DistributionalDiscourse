import argparse
import random
import logging
import os
from tqdm import tqdm

import data
import util

import torch
import torch.optim as optim
from pytorch_pretrained_bert import BertConfig, BertForMaskedLM, BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=str, 
        help='Name of the corpus file to train on (minus .json)')
parser.add_argument("--learning-rate", default=3e-5, type=float,
        help="The initial learning rate for Adam.")
parser.add_argument('--batch-size', type=int, default=50,
        help='Size of batches of utterances.')
parser.add_argument('--max-utt-len', type=int, default=50,
        help='Maximum utterance length in tokens (truncates first part of long utterances).')
parser.add_argument('--epochs', type=int, default=10,
        help='Number of times to iterate through the training data.')
parser.add_argument('-d','--data-dir', default='data',
        help='Data storage directory.')
parser.add_argument('-m','--model-dir', default='models',
        help='Trained model storage directory.')
parser.add_argument('--vocab-file', type=str, default='bert-base-uncased_vocab.txt', 
        help='Path of the customized BERT vocabulary to use.')
parser.add_argument('--cuda', action='store_true',
        help='use CUDA')
parser.add_argument('--gpu-id', type=int, default=0,
        help='Select with GPU to use')


def mask_text(tokens, tokenizer, vocab):
    masked_tokens, mask_indices = [], []
    for i, token in enumerate(tokens):
        if random.random() < 0.15:
            mask_indices.append(i)
            p = random.random()
            if p < .10:
                token = random.choice(vocab)
            elif p < .20:
                token = token
            else: 
                token = '[MASK]'
            masked_tokens.append(token)
            mask_indices.append(i)
        else:
            masked_tokens.append(token)
    masked_lm_labels = [-1] * len(tokens)
    for i in mask_indices:
        masked_lm_labels[i] = tokenizer.vocab[tokens[i]]
    return tokenizer.convert_tokens_to_ids(masked_tokens), masked_lm_labels


def gen_batches_masked_lm(data, batch_size, max_len, tokenizer):
    # flatten the data, ignoring the dialogue dimension
    vocab = list(tokenizer.vocab)
    x, y, i = [], [], 0
    for tokens in data:
        masked_tokens, masked_lm_labels = mask_text(tokens, tokenizer, vocab)
        x.append(masked_tokens)
        y.append(masked_lm_labels)
        i+= 1
        if i == batch_size:
            x = util.pad_lists(x, max_len)
            y = util.pad_lists(y, max_len)
            yield torch.LongTensor(x), torch.LongTensor(y)
            x, y, i = [], [], 0
    if x: # batch remainder
        x = util.pad_lists(x, max_len)
        y = util.pad_lists(y, max_len)
        yield torch.LongTensor(x), torch.LongTensor(y)


def train_epoch_masked_lm(bert_model, data, tokenizer, batch_size, max_len, optimizer, device):
    epoch_loss = 0
    batches = list(gen_batches_masked_lm(data, batch_size, max_len, tokenizer))

    for i, (x, y) in enumerate(tqdm(batches), 1): 
        x = x.to(device)
        y = y.to(device)
        loss = bert_model(x, masked_lm_labels=y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += (loss.item() / batch_size_) 
       
    return epoch_loss / i

if __name__ == '__main__':

    args = parser.parse_args()
    log = util.create_logger(logging.INFO)

    train_file = os.path.join(args.data_dir, f'{args.corpus}.json')
    vocab_file = os.path.join(args.data_dir, args.vocab_file)

    device = torch.device(f'cuda:{args.gpu_id}' if args.cuda and torch.cuda.is_available() else 'cpu')
    log.info(f"Training on {device}.")

    config = BertConfig.from_json_file('data/bert-base-uncased_config.json')
    bert_model = BertForMaskedLM(config)

    tokenizer = BertTokenizer.from_pretrained(vocab_file, 
            never_split=data.BERT_RESERVED_TOKENS + data.BERT_CUSTOM_TOKENS)
    train_data = data.load_data_pretraining(train_file, tokenizer)
    optimizer = optim.Adam(bert_model.parameters(), lr=args.learning_rate)
    

    for epoch in range(1, args.epochs+1):
        log.info(f"Starting epoch {epoch}")
        train_loss = train_epoch_masked_lm(bert_model, train_data, tokenizer, args.batch_size, args.max_utt_len, optimizer, device)
        log.info(f"Epoch {epoch} training loss:   {train_loss:.6f}")
        log.info(f"Saving epoch {epoch} model")
        bert_model.save_pretrained(args.model_dir, f'bert-base-uncased_MaskedLM-{args.corpus}.E{epoch}.bin')

