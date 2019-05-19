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

def damsl_tag_cluster(tag):
    """ DAMSL act clustering as described here:
    https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
    This is the same clustering used by Chen et. al., 2019
    Note both Jurafsky and Chen mention 42 tags, but Jurafsky lists 43.
    We find 43 tags with this method, + pad makes 44.
    """
    exceptions = {
        'qy^d': 'qy^d', # (Declarative yes-no Questions)
        'qw^d': 'qw^d', # (Declarative wh-questions) 
        'b^m' : 'b^m',  # (Signal-Understanding-via-Mimic)
        'nn^e': 'ng',
        'ny^e': 'na'}
    groups = [['qr', 'qy'], ['fe', 'ba'], ['oo', 'co', 'cc'], ['fx', 'sv'],
        ['fo', 'o', 'fw', '"', 'by', 'bc'], ['aap', 'am'] ,['arp', 'nd']]
    if '@' in tag:  # segmenting error (mark as pad; don't count towards loss/accuracy)
        return '@@@@@'
    for e in exceptions:
        if tag.startswith(e):
            return exceptions[e]
    if tag.startswith('^'):
        tag = '^' + re.split(r'[\^;,(]', tag)[1]
    else:
        tag = re.split(r'[\^;,(]', tag)[0]
    for group in groups:
        if tag in group:
            return '/'.join(group)
    return tag
