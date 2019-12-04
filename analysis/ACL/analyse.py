import sys
sys.path.append('../..')

from collections import defaultdict, namedtuple
import os
import data
import transformers
import argparse
import json

from eval_model import get_max_val_loss
from swda import swda
corpus = swda.CorpusReader('../../data/SWDA/swda')

parser = argparse.ArgumentParser()
parser.add_argument('corpus', choices=['SWDA', 'AMI-DA'],
        help='Which dialouge act corpus to analyse.')
parser.add_argument('-d','--data-dir', default='../../data',
        help='Data storage directory.')
args = parser.parse_args()

flatten = lambda l: [item for sublist in l for item in sublist]

def corpus_data():
    # All DA tags
    das = defaultdict(int)
    # DAs with laughter
    das_L = defaultdict(int)
    # DAs with laughter in the preceding utterance 
    das_preL = defaultdict(int)
    # DAs with laughter in the following utterance 
    das_postL = defaultdict(int)
    # DAs associated with laughter
    das_assocL = defaultdict(int)
    
    for trans in corpus.iter_transcripts():
        utts = [(u.damsl_act_tag(), u.text.lower()) for u in list(trans.utterances)]
        for pre, utt, post in zip(['@@@@@']+utts, utts, (utts+['@@@@@'])[1:]):
            das[utt[0]] += 1
            if '<laughter>' in utt[1]:
                das_L[utt[0]] += 1
            if '<laughter>' in " ".join([pre[1], utt[1], post[1]]):
                das_assocL[utt[0]] += 1
    return das, das_assocL, das_preL, das_L, das_postL

def test_data():
    test_file = os.path.join(args.data_dir, f'{args.corpus}_test.json')
    vocab_file = os.path.join(args.data_dir, f'bert-base-uncased_vocab.txt')
    tag_vocab_file = os.path.join(args.data_dir, f'{args.corpus}_tags.txt')
    tokenizer = data.load_tokenizer('bert-base-uncased')
    tokenizer = transformers.BertTokenizer.from_pretrained(vocab_file, 
                                                           never_split=data.BERT_RESERVED_TOKENS +\
                                                           data.BERT_CUSTOM_TOKENS)
    vocab_size = len(tokenizer.vocab)
    tag_vocab, tag2id = data.load_tag_vocab(tag_vocab_file)
    id2tag = dict((v,k) for k,v in tag2id.items())
    test_data = data.load_data(test_file, tokenizer, tag2id, strip_laughter="L")
    test_tags_ids = flatten([sample[1] for sample in test_data])
    test_tags_tags = [id2tag[t] for t in test_tags_ids]
    return test_tags_tags

def pred_data(url):
    with open(url) as f:
        pred_tt = json.load(f)
    return flatten([s for s in pred_tt])

def calculate_totals():
    das, das_assocL, das_preL, das_L, das_postL = corpus_data()
    # sort DAs by counter 
    das = sorted(das.items(), key=lambda x: x[1], reverse=True)
    totals = []
    for da in das:
        k = da[0]
        s = [k, da[1], das_assocL[k], das_preL[k], das_L[k], das_postL[k]]
        totals.append(s)
    return totals

def compute_accuracy_for_da(da, test, pred):
    comp = []
    for t, p in zip(test, pred):
        if t == da:
            res = 1 if t == p else 0
            comp.append([t, p, res])
    accuracy = lambda xs: sum([c[2] for c in xs ]) / len(xs)
    return accuracy(comp)

DialogueActStats = namedtuple('DialogueActStats', 
                              'nm name total total_ assoc_l assoc_l_ pre_l pre_l_ l l_ post_l post_l_ p1 p2')
class DialogueActStats(DialogueActStats):
    def __repr__(self):
        stats = list(self._asdict().items())[2:]
        vals = "|".join([str(round(v, 7)) for k,v in stats])
        return f'|{self.nm}|{self.name}|{vals}|'

if __name__ == '__main__':
    models_dir = '/scratch/DistributionalDiscourse/models/'
    names = json.load(open('SWDA_dialogue-acts.json'))
    totals = calculate_totals()
    total_c = sum([t[1] for t in totals])
    assoc_c = sum([t[2] for t in totals])
    pre_c = sum([t[3] for t in totals])
    l_c = sum([t[4] for t in totals])
    post_c = sum([t[5] for t in totals])

    test = test_data()
    model_dir1 = os.path.join(models_dir, 'SWDA-L_bert_2019-11-20')
    best_epoch1, _ = get_max_val_loss(model_dir1)
    pred2 = pred_data(os.path.join(model_dir1, f'preds.E{best_epoch1}.json'))
    model_dir2 = os.path.join(models_dir, 'SWDA-NL_bert_2019-11-20')
    best_epoch2, _ = get_max_val_loss(model_dir1)
    pred1 = pred_data(os.path.join(model_dir1, f'preds.E{best_epoch1}.json'))
    stats = []
    for t in totals:
        total_ = t[1] / total_c
        assoc_l_ = t[2] / t[1]
        pre_l_ = t[3] / t[1]
        l_ = t[4] / t[1]
        post_l_ = t[5] / t[1]
        p1acc = compute_accuracy_for_da(t[0], test, pred1)
        p2acc = compute_accuracy_for_da(t[0], test, pred2)
        stats.append(DialogueActStats(t[0], names[t[0]], t[1], total_, t[2], assoc_l_,
                                      t[3], pre_l_, t[4], l_, t[5], post_l_, p1acc, p2acc))
    for s in stats:
        print(s)

## Local Variables:
## python-shell-interpreter: "../../nix-shell"
## python-shell-interpreter-args: "--run python"
## End:
    
