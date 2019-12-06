import sys
sys.path.append("../..") # Adds package base dir to python modules path.
from eval_model import get_max_val_loss
import json
import pandas as pd
from collections import defaultdict
import os


def gen_model_preds_df(corpus, models, model_dirs,group=False):
    """
    corpus - one of: 'SWDA-L', 'SWDA-NL', 'AMI-L', 'AMI-NL'
    """
    with open(f'../../data/{corpus}_test.json') as f:
        test_data = json.load(f)
    with open(f'{corpus}_dialogue-acts-groups.json') as f:
        groups = json.load(f)
        groups[None] = 'Other'
    if not group:
        test_data = [{
            'dialogue_id': d['id'],
            'utt_no': i,
            'speaker':speaker,
            'utt': utt,
            'da_tag': da_tag} for d in test_data
                     for i, (speaker, utt, da_tag)
                     in enumerate(zip(d['speakers'], d['utts'], d['da_tags']))]
    else:
        test_data = [{
            'dialogue_id': d['id'],
            'utt_no': i,
            'speaker':speaker,
            'utt': utt,
            'da_tag': groups[da_tag]} for d in test_data
                     for i, (speaker, utt, da_tag)
                     in enumerate(zip(d['speakers'], d['utts'], d['da_tags']))]
        

    df = pd.DataFrame(test_data)
    df = df.set_index(['dialogue_id', 'utt_no'])
    
    preds = defaultdict(lambda x: dict)
    for model, model_dir in zip(models, model_dirs):
        best_epoch, _ = get_max_val_loss(model_dir)
        preds_file = os.path.join(model_dir, f'preds.E{best_epoch}.json')
        with open(os.path.join(model_dir, preds_file), 'r') as f:
            preds = json.load(f)
            if not group:
                preds = [((u['dialogue_id'], u['utt_no']), pred_da)
                         for pred_da, u in zip([da for d in preds for da in d], test_data)]
            else:
                preds = [((u['dialogue_id'], u['utt_no']), pred_da)
                         for pred_da, u in zip([groups[da] for d in preds for da in d], test_data)]
            preds = list(zip(*preds))
            preds = pd.Series(preds[1], index=preds[0])
            df[model] = preds

    return df
