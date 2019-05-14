from swda.swda import CorpusReader

from preproc import tokenize
from pytorch_pretrained_bert.tokenization import PRETRAINED_VOCAB_ARCHIVE_MAP, BertTokenizer, load_vocab as load_bert_vocab 

import os
import re
import json
import zipfile
import random
import argparse
import urllib.request
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=['prep-swda', 'download-glove', 'customize-bert-vocab'], help="What to process")

SWDA_CORPUS_DIR = "data/swda"
SWDA_SPLITS = "data/swda_{}.json"

BERT_MODEL = 'bert-base-uncased'
BERT_VOCAB_FILE = "data/{}-vocab.SWDA.txt"
BERT_RESERVED_TOKENS = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"] # used by the pre-trained BERT model
BERT_CUSTOM_TOKENS = ['[SPKR_A]', '[SPKR_B]', '<laughter>'] # added by us TODO: add disfluencies

vocab =  {"@@@@@":0, "[SPKR_A]":1, "[SPKR_B]":2}
tag_vocab = {}

def gen_splits(id_list, train=0.7, val=0.1, test=0.2):
    assert(train+val+test == 1)
    random.shuffle(id_list)
    n_train, n_val, n_test = [int(x * len(id_list)) for x in (train, val, test)]
    train, val, test = id_list[:n_train], id_list[n_train:n_train+n_val], id_list[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test} 

def load_vocab(vocab_file):
    with open(vocab_file) as f:
        token2id = json.load(f) # token -> int dictionary
    id2token = [item[0] for item in sorted(token2id.items(), key=lambda x: x[1])]
    return (id2token, token2id)

def load_glove(glove_dim, vocab):
    with open('data/glove.6B/glove.6B.{}d.txt'.format(glove_dim), 'rb') as f:
        word_vectors = {}
        for line in tqdm(f.readlines(), desc="loading glove {}d".format(glove_dim)):
            word_vectors[line[0]] = list(map(float, line[1:]))
    # order the word vectors according to the vocab
    word_vectors = [word_vectors[w] if w in word_vectors else [0] * glove_dim for w in vocab]
    return word_vectors

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def prep_swda():
    """
    Put the conversations into a json format that torchtext can read easily.
    Each "example" is a conversation comprised of a list of utterances 
    and a list of dialogue act tags (each the same length)
    """

    print("Loading SWDA corpus.")
    if not os.path.isfile(SWDA_CORPUS_DIR):
        with zipfile.ZipFile("swda/swda.zip") as zip_ref:
            zip_ref.extractall('data')
    corpus = CorpusReader(SWDA_CORPUS_DIR)
    corpus = {t.conversation_no: t for t in corpus.iter_transcripts()}

    bert_vocab_file = BERT_VOCAB_FILE.format(BERT_MODEL)
    if not os.path.isfile(bert_vocab_file):
        print("Customizing BERT vocab.")
        customize_bert_vocab()
    print("Loading BERT vocab/tokenizer.")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_vocab_file, 
            never_split = BERT_RESERVED_TOKENS + BERT_CUSTOM_TOKENS)

    print("Getting splits.")
    splits_file = SWDA_SPLITS.format('splits')
    if os.path.isfile(splits_file): # use existing SWDA splits (for reproducibility purposes)
        with open(splits_file) as f:
            splits = json.load(f)
    else: # save the splits file
        splits = gen_splits(list(corpus.keys()))
        with open(splits_file, 'w') as f:
            json.dump(splits, f)

    def words_to_ints(ws):
        maxvalue = max(vocab.values())
        for w in ws:
            if w not in vocab:
                maxvalue += 1
                vocab[w] = maxvalue
        xs = [vocab[x] for x in ws]
        return xs

    def tag_to_int(tag):
        maxvalue = max(tag_vocab.values()) if tag_vocab else 0
        if tag not in tag_vocab:
            maxvalue += 1
            tag_vocab[tag] = maxvalue
        return tag_vocab[tag] 

    def extract_example(transcript):
        """ Gets the parts we need from the SWDA utterance object """ 
        tags, tags_ints, utts, utts_ints, utts_ints_bert = [], [], [], [], []
        for utt in transcript.utterances:
            # Regex tokenization
            words = "[SPKR_{}] ".format(utt.caller) + tokenize(utt.text.lower())
            utts.append(words)
            utts_ints.append(words_to_ints(words.split()))
            # BERT wordpiece tokenization
            bert_text = "[CLS] [SPKR_{}] ".format(utt.caller) + utt.text
            bert_tokens = bert_tokenizer.tokenize(bert_text) # list of strings
            utts_ints_bert.append(bert_tokenizer.convert_tokens_to_ids(bert_tokens))
            # dialogue act tags
            tags.append(utt.act_tag)
            tags_ints.append(tag_to_int(utt.act_tag))
        return {'id': transcript.conversation_no, 'utts': utts, 'utts_ints': utts_ints, 
                'utts_ints_bert': utts_ints_bert, 'tags': tags, 'tags_ints': tags_ints}

    print("Extracting data and saving splits.")
    for split in splits:
        data = []
        for ex_id in tqdm(splits[split], desc=split):
            data.append(extract_example(corpus[ex_id]))
        with open(SWDA_SPLITS.format(split), 'w') as f:
            json.dump(data, f)
    with open(SWDA_SPLITS.format("vocab"), 'w') as f:
        json.dump(vocab, f)
    with open(SWDA_SPLITS.format("tag_vocab"), 'w') as f:
        json.dump(tag_vocab, f)


def customize_bert_vocab():
    vocab_filename = BERT_VOCAB_FILE.format(BERT_MODEL) 
    vocab_url = PRETRAINED_VOCAB_ARCHIVE_MAP[BERT_MODEL]
    download_url(vocab_url, vocab_filename)
    vocab = list(load_bert_vocab(vocab_filename).keys()) # load_vocab gives an OrderedDict 
    custom_tokens = ['[SPKR_A]', '[SPKR_B]', '<laughter>'] # TODO: add disfluencies
    # most of the first 1000 tokens are [unusedX], but [PAD], [CLS], etc are scattered in there too 
    for new_token in custom_tokens:
        for i, existing_token in enumerate(vocab):
            if re.match(r"\[unused\d+\]", existing_token):
                vocab[i] = new_token
                print("Custom BERT vocab: {} -> {} (replaced {})".format(new_token, i, existing_token))
                break
            elif i > 1000:
                raise ValueError("Couldn't find any unused tokens to replace :(")
    with open(vocab_filename, 'w', encoding="utf-8") as f:
        for token in vocab:
            f.write(token + '\n')

def download_glove():
    glove_file = 'data/glove.6B.zip'
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    if not os.path.isfile(glove_file): 
        download_url(glove_url, 'data/glove.6B.zip')
    with  zipfile.ZipFile(glove_file, 'r') as zip_ref:
        zip_ref.extractall('data/glove.6B')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.command == 'prep-swda':
        prep_swda()
    if args.command == 'download-glove':
        download_glove()
    if args.command == 'customize-bert-vocab':
        customize_bert_vocab()

        
