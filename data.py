import argparse
import logging
import os
import json
import csv
import zipfile
import tarfile
from collections import namedtuple
from tqdm import tqdm

import util
import ami
from swda import swda

from preproc import tokenize, damsl_tag_cluster, remove_laughters, remove_disfluencies

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=['prep-corpora', 'customize-bert-vocab'], 
        help="What preprocessing to do.")
parser.add_argument('-c','--corpora', nargs='+', default=[], 
        help='A list of corpora to preprocess. (Default: preprocess all)'
             'Options: swbd, swda, ami, ami-da')
parser.add_argument('-d','--data-dir', default='data',
        help='Data storage directory.')
parser.add_argument('-pt','--pause-threshold', default=1,
        help='Threshold for heuristically segementing same-speaker utterances (seconds).')


def customize_bert_vocab(bert_model='bert-base-uncased'):

    from pytorch_pretrained_bert.tokenization import PRETRAINED_VOCAB_ARCHIVE_MAP, BertTokenizer, load_vocab
    bert_vocab_file = os.path.join(DATA_DIR, f"{bert_model}-vocab.txt")

    vocab_filename = BERT_VOCAB_FILE.format(bert_model) 
    vocab_url = PRETRAINED_VOCAB_ARCHIVE_MAP[bert_model]
    util.download_url(vocab_url, vocab_filename)
    vocab = list(load_vocab(vocab_filename).keys()) # load_vocab gives an OrderedDict 
    custom_tokens = ['[SPKR_A]', '[SPKR_B]', '<laughter>'] # TODO: add disfluencies
    # most of the first 1000 tokens are [unusedX], but [PAD], [CLS], etc are scattered in there too 
    for new_token in custom_tokens:
        for i, existing_token in enumerate(vocab):
            if re.match(r"\[unused\d+\]", existing_token):
                vocab[i] = new_token
                log.info("Custom BERT vocab: {} -> {} (replaced {})".format(new_token, i, existing_token))
                break
            elif i > 1000:
                raise ValueError("Couldn't find any unused tokens to replace :(")
    with open(vocab_filename, 'w', encoding="utf-8") as f:
        for token in vocab:
            f.write(token + '\n')


Dialogue = namedtuple('SegmentedDialogue', ['id', 'speakers', 'utts', 'da_tags'])


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

def parse_ami(corpus_dir, pause_threshold):

    log.info("Loading the AMI corpus...")
    ami_meetings = ami.get_corpus(corpus_dir, da_only=False)

    dialogues_da, dialogues_noda = [], []
    for m in tqdm(ami_meetings):
        m.gen_transcript(utt_pause_threshold=pause_threshold)
        if m.speaker_dialogue_acts:  # DA-tagged meeting
            speakers, utts, da_tags = [], [], []
            for utt in m.transcript:
                speakers.append(utt.speaker)
                utts.append(utt.text())
                da_tags.append(utt.dialogue_act.da_tag)
            dialogues_da.append(Dialogue(m.meeting_id, speakers, utts, da_tags))
        else:  # not DA-tagged meeting
            speakers, utts = [], []
            for utt in m.transcript:
                speakers.append(utt.speaker)
                utts.append(utt.text())
            dialogues_noda.append(Dialogue(m.meeting_id, speakers, utts, None))
    return dialogues_da, dialogues_noda


# def parse_swbd(corpus_dir):
    # log.info("Parsing SWBD.")
    # fieldnames = ['id', 'treebank_id', 'start_word', 'end_word', 'alignment_tag',
            # 'ldc_word', 'ms98_word']
    # words_dir = os.path.join(corpus_dir, 'data', 'alignments')
    # swbd_files = [os.path.join(words_dir, subdir, f)
                    # for subdir in os.listdir(words_dir) 
                    # for f in os.listdir(os.path.join(words_dir, subdir))]
    # dialogue_ids = list({os.path.basename(f)[:6] for f in swbd_files})
    # dialogues = []
    # for dialogue_id in dialogue_ids:
        # speakers, tokens = [], [] 
        # dialogue_files = [f for f in swbd_files if os.path.basename(f).startswith(dialogue_id)]
        # for filename in dialogue_files:
            # speaker = os.path.basename(filename)[6]
            # with open(filename) as f:
                # reader = csv.DictReader(f, fieldnames, delimiter='\t')
                # for line in reader:
                    # tokens.append(line['ldc_word'])
                    # speakers.append(speaker)
        # dialogues.append(UnsegmentedDialogue(dialogue_id, speakers, tokens))
    # return dialogues


def parse_swda(corpus_dir):
    log.info("Parsing SWDA.")
    corpus = swda.CorpusReader(os.path.join(corpus_dir,'swda'))
    dialogues = []
    for transcript in corpus.iter_transcripts():
        speakers, utts, da_tags = [], [], []
        for utt in transcript.utterances:
            speakers.append(utt.caller)
            utts.append(utt.text)
            da_tags.append(utt.damsl_act_tag())  # Utterance.damsl_act_tag implements clustering
        dialogue_id = 'sw' + str(transcript.conversation_no)
        dialogues.append(Dialogue(dialogue_id, speakers, utts, da_tags))
    return dialogues


def write_corpus(dialogues, data_dir, filename):
    path = os.path.join(data_dir, filename)
    dialogues = [d._asdict() for d in dialogues]
    with open(path, 'w') as f:
        json.dump(dialogues, f)
    log.info(f"Wrote {len(dialogues)} dialogues to {filename}.")


if __name__ == '__main__':

    args = parser.parse_args()
    log = util.create_logger(logging.INFO)

    ami_url = "http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
    # swbd_url = "http://www.isip.piconepress.com/projects/switchboard/releases/ptree_word_alignments.tar.gz"

    ami_zip_file = os.path.join(args.data_dir, os.path.basename(ami_url))
    # swbd_zip_file = os.path.join(args.data_dir, os.path.basename(swbd_url))
    swda_zip_file = "swda/swda.zip"  # included in the swda sub-module

    ami_dir = os.path.join(args.data_dir, 'AMI')
    # swbd_dir = os.path.join(args.data_dir, 'SWBD')
    swda_dir = os.path.join(args.data_dir, 'SWDA')

    if args.command == 'prep-corpora':


        download_corpus(ami_url, ami_zip_file)
        # download_corpus(swbd_url, swbd_zip_file)

        extract_corpus(ami_zip_file, ami_dir)
        # extract_corpus(swbd_zip_file, swbd_dir)
        extract_corpus(swda_zip_file, swda_dir)

        ami_da, ami_noda = parse_ami(ami_dir, args.pause_threshold)
        # swbd = parse_swbd(swbd_dir)
        swda = parse_swda(swda_dir)

        write_corpus(ami_da, args.data_dir,  'AMI-DA.json')
        write_corpus(ami_noda, args.data_dir, 'AMI-noDA.json')
        write_corpus(swda, args.data_dir , 'SWDA.json')
        # write_corpus(swbd, args.data_dir, 'SWBD.unsegmented.json')

    if args.command == 'customize-bert-vocab':
        pass

