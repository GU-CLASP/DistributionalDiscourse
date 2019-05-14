import re

def remove_laughters(s):
    s = re.sub(r'<laughter>', '', s)
    return s

def remove_disfluencies(s):
    s = re.sub(r'{\w', '', s)
    s = re.sub(r'}', '', s)
    return s

def tokenize(s,disfluencies=True,laughters=True):
    s = re.sub(r'(\-)(?=\W)', r' \1', s)
    s = re.sub(r'([,.?!])', r' \1', s) 
    s = re.sub(r'(n\'t)', r' \1',s)
    s = re.sub(r'(\'re|\'s|\'m|\'d)', r' \1',s)
    s = re.sub(r'\s(?=[\w\s]+>>)', '_',s)
    if not laughters:
        s = remove_laughters(s)
    if not disfluencies:
        s = remove_disfluencies(s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def bert_tokenize(bert_tokenizer, s, disfluencies=True, laughters=True):
    """ bert_tokenizer is a pytorch_pretrained_bert.BertTokenizer
    """
    if not laughters:
        s = remove_laughters(s)
    if not disfluencies(s):
        s = remove_disfluencies(s)
    return s

