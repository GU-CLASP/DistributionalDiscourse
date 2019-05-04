# import torch
# import torchtext as tt
from swda.swda import CorpusReader
from preproc import tokenize
import os
import json
import zipfile
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("step", choices=['prep-swda'], help="The preprocessing step")
args = parser.parse_args()

SWDA_CORPUS_DIR = "data/swda"
SWDA_SPLITS = "data/swda_{}.json"

vocab =  {"@@@@@":0, "spkr_A":1, "spkr_B":2}

def gen_splits(id_list, train=0.7, val=0.1, test=0.2):
    assert(train+val+test == 1)
    random.shuffle(id_list)
    n_train, n_val, n_test = [int(x * len(id_list)) for x in (train, val, test)]
    train, val, test = id_list[:n_train], id_list[n_train:n_train+n_val], id_list[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test} 

def prep_swda():
    """
    Put the conversations into a json format that torchtext can read easily.
    Each "example" is a conversation comprised of a list of utterances 
    and a list of dialogue act tags (each the same length)
    """

    if not os.path.isfile(SWDA_CORPUS_DIR):
        with zipfile.ZipFile("swda/swda.zip") as zip_ref:
            zip_ref.extractall('data')

    corpus = CorpusReader(SWDA_CORPUS_DIR)
    corpus = {t.conversation_no: t for t in corpus.iter_transcripts()}

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

    def extract_example(transcript):
        """ Gets the parts we need from the SWDA utterance object """ 
        tags, utts, utts_ints = [], [], []
        for utt in transcript.utterances:
            words = "spkr_{} ".format(utt.caller) + tokenize(utt.text.lower())
            ints = words_to_ints(words.split())
            utts.append(words)
            utts_ints.append(ints)
            tags.append(utt.act_tag)
        return {'id': transcript.conversation_no, 'utts': utts, 'utts_ints': utts_ints, 'tags': tags}

    splits = {split: [extract_example(corpus[ex_id]) for ex_id in splits[split]] for split in splits}
    for split in splits:
        with open(SWDA_SPLITS.format(split), 'w') as f:
            json.dump(splits[split], f)
    with open(SWDA_SPLITS.format("vocab"), 'w') as f:
        json.dump(vocab, f)

if __name__ == '__main__':
    if args.step == 'prep-swda':
        prep_swda()

        
