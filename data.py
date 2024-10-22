import argparse
import logging
import os
import json
import csv
import re
import zipfile
import tarfile
import random
from collections import namedtuple, defaultdict
from tqdm import tqdm

import util
import ami
from swda import swda

import transformers 

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=[
    'prep-corpora', 
    'download-glove', 
    'get-bert-config',
    'test-tokenization',
    'prep-pretraining-corpora',
    'prep-opensubtitles-pretraining-corpus'
    ],  help="What preprocessing to do.")
parser.add_argument('-d','--data-dir', default='data',
        help='Data storage directory.')
parser.add_argument('-pt','--pause-threshold', default=1,
        help='Threshold for heuristically segementing same-speaker utterances (seconds).')

TAG_PAD = '@@@@@'  # also used for unknown tags (not counted in loss)
SPEAKER_TOKENS = [f'[SPKR_{a}]' for a in ['A', 'B', 'C', 'D', 'E']] 
LAUGHTER_TOKEN = '<laughter>'
BERT_CUSTOM_TOKENS = SPEAKER_TOKENS + [LAUGHTER_TOKEN]
BERT_RESERVED_TOKENS = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"] # used by the pre-trained BERT model


def load_tokenizer(bert_model):
    tokenizer = transformers.BertTokenizer.from_pretrained(bert_model)
    tokenizer.add_tokens(BERT_CUSTOM_TOKENS)
    return tokenizer


def load_glove(data_dir, glove_dim, tokenizer, log=None):
    # TODO: this can deal with word pieces better...
    glove_dir = os.path.join(data_dir, 'glove.6B')
    if not os.path.exists(glove_dir):
        download_glove(data_dir)
    glove_file = os.path.join(glove_dir, f'glove.6B.{glove_dim}d.txt')
    if not os.path.exists(glove_file):
        raise ValueError(f"We don't have {glove_dim}-dimensional gloVe vectors.")
    with open(glove_file, 'r') as f:
        word_vectors = {}
        for line in tqdm(f.readlines(), desc="loading glove {}d".format(glove_dim)):
            line = line.strip().split(' ')
            w = str(line[0])
            v = [float(vi) for vi in line[1:]]
            word_vectors[w] = v
    vocab = dict(tokenizer.vocab, **tokenizer.added_tokens_encoder)
    weights = [[random.uniform(-1,1) for i in range(glove_dim)] for j in range(len(vocab))]
    n = 0
    for w in vocab:
        if w in word_vectors:
            weights[vocab[w]] = word_vectors[w]
            n += 1
    if log:
        log.info(f"Initialized {n} of {len(vocab)} words with GloVe")

    return weights 


def download_glove(data_dir):
    glove_file = os.path.join(data_dir, 'glove.6B.zip')
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    if not os.path.isfile(glove_file):
        util.download_url(glove_url, glove_file)
    with  zipfile.ZipFile(glove_file, 'r') as zip_ref:
        zip_ref.extractall('data/glove.6B')


def load_data(corpus_file, tokenizer, tag2id, strip_laughter=False, tag_field='da_tags', ignore_tags=[]):
    with open(corpus_file) as f:
        dialogs = json.load(f)
    data = []
    for d in dialogs:
        utts, tags = [], []
        for speaker,utt,tag in zip(d['speakers'], d['utts'], d[tag_field]):
            if tag in ignore_tags:
                tag = TAG_PAD
            utt = [f'[SPKR_{speaker}]'] + tokenizer.tokenize(utt)
            if strip_laughter:
                utt = [t for t in utt if t != LAUGHTER_TOKEN]
                if not utt:  # laughter is the only token; skip this utt
                    continue
            encoded_utt = tokenizer.encode(utt, add_special_tokens=True)
            utts.append(encoded_utt)
            tags.append(tag2id.get(tag, 0))
        data.append((utts, tags))
    return data


def prep_dialog_for_pretraining(dialog):
    utts = []
    for speaker,utt in zip(dialog['speakers'], dialog['utts']):
       utt = f'[SPKR_{speaker}] ' + utt 
       utts.append(utt)
    return utts


def load_data_pretraining(corpus_file):
    with open(corpus_file) as f:
        dialogs = json.load(f)
    data = []
    for d in dialogs:
        data.append(prep_dialog_for_pretraining(d))
    return data


def write_pretraining_corpus(corpus_file, corpus):
    with open(os.path.join(corpus_file), 'w') as f:
        for d in corpus:
            for utt in d:
                f.write(utt + '\n')
            f.write('\n')

Dialogue = namedtuple('Dialogue', ['id', 'speakers', 'utts', 'da_tags', 'laugh_types'])


def extract_corpus(zip_file, corpus_dir):
    if os.path.exists(corpus_dir):
        log.info(f"The corpus directory, {corpus_dir} alrdeay exists.")
        return
    log.info(f"Extracting {zip_file} to {corpus_dir}.")
    if zip_file.endswith('.zip'):
        file_handler = lambda x: zipfile.ZipFile(x, 'r')
    elif zip_file.endswith('tar.gz'):
        file_handler = lambda x: tarfile.open(x, 'r:gz')
    else:
        raise ValueError(f"Can't handle extension {zip_ext} when unzipping {zip_file}.")
    with file_handler(zip_file) as f:
        f.extractall(corpus_dir)

def download_corpus(url, zip_file):
    if os.path.exists(zip_file):
        log.info(f"Skipping download of {zip_file} (already exists).")
        return
    util.download_url(url, zip_file)

def laugh_type(utt):
    laugh = '<laughter>'
    utt = re.sub(r'[#.\-,]','',utt).strip().lower()
    if utt == laugh:
        return 'stand alone'
    elif laugh in utt:
        if utt.startswith(laugh):
            return 'prefix'
        elif utt.endswith(laugh):
            return 'suffix'
        else:
            return 'infix'
    else:
        return 'none'

def normalize_ami(utt):
    s = str(utt) # AMIUtterance to string defined in ami.py
    # Remove disfluencies and vocal sounds other than laughter standardly: 
    #    <other>, <cough>, but also custom trascriber tokens  which might 
    #    include whitespace and hyphens (e.g., <imitates zapping> (IS1007b.D),
    #    or <long sh- sound> (ISI1004.A))
        
    s = re.sub(r'([A-Z])_', r'\1 ', s) # acronyms: D_V_D_ -> D V D 
    s = re.sub(r'((?!<laugh>)<[\w -]+>)', '', s)
    # normalize the laughter token
    s = re.sub(r'<laugh>', '<laughter>', s)
    return s

def parse_ami(corpus_dir, pause_threshold):

    log.info("Parsing AMI...")
    ami_meetings = ami.get_corpus(corpus_dir, da_only=False)

    dialogs_da, dialogs_noda = [], []
    for m in tqdm(ami_meetings):
        m.gen_transcript(utt_pause_threshold=pause_threshold)
        if m.speaker_dialog_acts:  # DA-tagged meeting
            speakers, utts, da_tags, laugh_types = [], [], [], []
            for utt in m.transcript:
                speakers.append(utt.speaker)
                utts.append(normalize_ami(utt))
                da_tags.append(utt.dialog_act.da_tag)
                laugh_types.append(laugh_type(str(utt)))
            dialogs_da.append(Dialogue(m.meeting_id, speakers, utts, da_tags, laugh_types))
        else:  # not DA-tagged meeting
            speakers, utts, laugh_types = [], [], []
            for utt in m.transcript:
                speakers.append(utt.speaker)
                utts.append(normalize_ami(utt))
                laughter_types.append(laughter_type(str(utt)))
            laughter_type_next = laughter_types[1:] + ['none'] # predict the laughter type of the next utterance. for the last utterance, use 'none' 
            dialogs_noda.append(Dialogue(m.meeting_id, speakers, utts, None, laughter_type_next))
    return dialogs_da, dialogs_noda


def normalize_swda(s):

    def remove_disfluencies(s):
        s = re.sub(r'{\w', '', s)
        s = re.sub(r'}', '', s)
        return s

    s = re.sub(r'\*\[\[.+\]\]', '', s) # annotator comments: *[[listen; is this a question?]]
    s = re.sub(r'\[(.+?)\+(.+?)\]', r'\1 \2', s) # repairs: [a, + a] -> a a 
    s = re.sub(r'(\-)(?=\W)', r' \1', s)
    s = re.sub(r'([,.?!])', r' \1', s) 
    # s = re.sub(r'(n\'t)', r' \1',s) 
    # s = re.sub(r'(\'re|\'s|\'m|\'d)', r' \1',s)
    s = re.sub(r'\s(?=[\w\s]+>>)', '_',s)
    s = remove_disfluencies(s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'((?!<[lL]aughter>)<[\w_]+>)', '', s)
    s = re.sub(r'<Laughter>', '<laughter>', s) # normalize the laughter token
    s = re.sub(r' */ *$', '', s) # remove the slash at the end of lines
    s = s.strip()

    return s

def parse_swda(corpus_dir):
    log.info("Parsing SWDA...")
    corpus = swda.CorpusReader(os.path.join(corpus_dir,'swda'))
    dialogs = []
    for transcript in corpus.iter_transcripts():
        speakers, utts, da_tags, laugh_types = [], [], [], []
        for utt in transcript.utterances:
            speakers.append(utt.caller)
            utts.append(normalize_swda(utt.text))
            da_tags.append(utt.damsl_act_tag())  # Utterance.damsl_act_tag implements clustering
            laugh_types.append(laugh_type(utt.text))
        dialog_id = 'sw' + str(transcript.conversation_no)
        dialogs.append(Dialogue(dialog_id, speakers, utts, da_tags, laugh_types))
    return dialogs


def write_tag_vocab(dialogs, data_dir, corpus_name):
    tag_file = os.path.join(data_dir, f'{corpus_name}_tags.txt')
    if os.path.exists(tag_file):
        log.info(f"Tag vocab already exsits at {tag_file}.")
        return
    tags = list({t for d in dialogs for t in d.da_tags if t})
    tags = [TAG_PAD] + tags
    with open(tag_file, 'w') as f:
        for tag in tags:
            f.write(tag + '\n')
    log.info(f"Wrote {len(tags)} to {tag_file}.")


def load_tag_vocab(vocab_file):
    tag_vocab = []
    with open(vocab_file, 'r') as f:
        for tag in f.readlines():
            tag = tag.strip()
            tag_vocab.append(tag)
    tag2id = {tag: i for i, tag in enumerate(tag_vocab)}
    tag2id = defaultdict(lambda: tag2id[TAG_PAD], tag2id)
    return tag_vocab, tag2id


def write_corpus(dialogs, data_dir, filename):
    path = os.path.join(data_dir, filename)
    dialogs = [d._asdict() for d in dialogs]
    with open(path, 'w') as f:
        json.dump(dialogs, f)
    log.info(f"Wrote {len(dialogs)} dialogs to {filename}.")


def gen_splits(ids, train=0.7, val=0.1, test=0.2):
    assert(train+val+test == 1)
    ids = list(ids)
    random.shuffle(ids)
    n_train, n_val, n_test = [int(x * len(ids)) for x in (train, val, test)]
    train, val, test = ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test}


def write_splits(dialogs, data_dir, corpus_name):
    splits_file = os.path.join(data_dir, f"{corpus_name}_splits.json")
    if os.path.exists(splits_file):
        log.info(f"Loading splits from {splits_file}.")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
    else:
        log.info(f"Generating splits and saving to {splits_file}.")
        splits = gen_splits([d.id for d in dialogs])
        with open(splits_file, 'w') as f:
            json.dump(splits, f)
    dialogs = {d.id: d for d in dialogs}
    for split in splits:
        split_data = [dialogs[dialog_id] for dialog_id in splits[split]]
        write_corpus(split_data, data_dir, f"{corpus_name}_{split}.json")


if __name__ == '__main__':

    log = util.create_logger(logging.INFO)
    args = parser.parse_args()

    ami_url = "http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
    # swbd_url = "http://www.isip.piconepress.com/projects/switchboard/releases/ptree_word_alignments.tar.gz"

    ami_zip_file = os.path.join(args.data_dir, os.path.basename(ami_url))
    # swbd_zip_file = os.path.join(args.data_dir, os.path.basename(swbd_url))
    swda_zip_file = "swda/swda.zip"  # included in the swda sub-module

    ami_dir = os.path.join(args.data_dir, 'AMI')
    # swbd_dir = os.path.join(args.data_dir, 'SWBD')
    swda_dir = os.path.join(args.data_dir, 'SWDA')

    swda_splits_file = os.path.join(args.data_dir, 'SWDA_splits.json')
    ami_splits_file = os.path.join(args.data_dir, 'AMI-DA_splits.json')

    if args.command == 'prep-corpora':

        # AMI (with/without dialog acts)
        download_corpus(ami_url, ami_zip_file)
        extract_corpus(ami_zip_file, ami_dir)
        ami_da, ami_noda = parse_ami(ami_dir, args.pause_threshold)
        write_tag_vocab(ami_da, args.data_dir, 'AMI-DA')
        write_splits(ami_da, args.data_dir, 'AMI-DA')
        write_corpus(ami_noda, args.data_dir, 'AMI-noDA.json')

        # Switchboard Dialogue Act Corpus
        extract_corpus(swda_zip_file, swda_dir)
        swda = parse_swda(swda_dir)
        write_tag_vocab(swda, args.data_dir, 'SWDA')
        write_splits(swda, args.data_dir, 'SWDA')
       
        # Switchboard 
        # download_corpus(swbd_url, swbd_zip_file)
        # extract_corpus(swbd_zip_file, swbd_dir)
        # swbd = parse_swbd(swbd_dir)
        # write_corpus(swbd, args.data_dir, 'SWBD.unsegmented.json')


    if args.command == 'get-bert-config':
        """
        Really silly, but we need to save the config from the pre-trained BERT so we can replicate it 
        when we're using the randomly-initalized BERT.
        """
        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        bert.config.to_json_file(os.path.join(args.data_dir, 'bert-base-uncased_config.json'))

    if args.command == 'test-tokenization':
        corpus = 'AMI-DA'
        tokenizer = load_tokenizer('bert-base-uncased')
        tag_vocab, tag2id = load_tag_vocab(f'data/{corpus}_tags.txt')
        data = load_data(f'data/{corpus}_train.json', tokenizer, tag2id)
        with open (f'data/{corpus}_train.json') as f:
            dialogs = json.load(f)
        for (x, y), d in zip(data, dialogs):
            print(f"Dialogue {d['id']}")
            for utt, tag, utt_d, tag_d in zip(x,y,d['utts'],d['da_tags']):
                utt = tokenizer.convert_ids_to_tokens(utt)
                tag = tag_vocab[tag]
                print(f"{tag: <12} {' '.join(utt)}")
                print(f"{tag_d if tag_d else 'None': <12}          {utt_d}")
            print()

    if args.command == 'download-glove':
        download_glove(args.data_dir)

    if args.command == 'prep-pretraining-corpora':

        ami = load_data_pretraining(os.path.join(args.data_dir, 'AMI-DA_train.json'))
        ami_da = load_data_pretraining(os.path.join(args.data_dir, 'AMI-noDA.json'))
        swda = load_data_pretraining(os.path.join(args.data_dir, 'SWDA_train.json'))

        write_pretraining_corpus(os.path.join(args.data_dir, 'AMI_pretraining.txt'), ami + ami_da)
        write_pretraining_corpus(os.path.join(args.data_dir, 'SWBD_pretraining.txt'), swda)
        write_pretraining_corpus(os.path.join(args.data_dir, 'AMI+SWBD_pretraining.txt'), ami + ami_da + swda)

    if args.command == 'prep-opensubtitles-pretraining-corpus':

        from opensubtitles import os_to_json

        def iter_dialogs(handle, ids_logfile, limit_utts=None):
            total_utts = 0
            pbar = tqdm(total=limit_utts)
            for dialog, n_laughters in handle:
                dialog_len = len(dialog['utts'])
                if (n_laughters / dialog_len) < 0.01:
                    continue
                total_utts += dialog_len
                if total_utts >= limit_utts:
                    break
                else:
                    pbar.update(dialog_len)
                ids_logfile.write(dialog['id']+'\n')
                yield prep_dialog_for_pretraining(dialog)
            pbar.close()

        corpus_file = os.path.join(args.data_dir, "OS_pretraining.txt")
        corpus_log  = os.path.join(args.data_dir, "OS_ids.txt")
        handle = os_to_json(
            '/scratch/DistributionalDiscourse/opensubtitles/OpenSubtitles/xml/en/',
            shuffle=True
        )

        with open(corpus_log, 'w', buffering=1) as ids_logfile:
            write_pretraining_corpus(corpus_file, iter_dialogs(handle, ids_logfile, limit_utts=int(1e+8)))
