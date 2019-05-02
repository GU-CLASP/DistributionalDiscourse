import re

def tokenize(s,disfluencies=True,laughters=True):
    s = re.sub(r'(\-)(?=\W)', r' \1', s)
    s = re.sub(r'([,.?!])', r' \1', s) 
    s = re.sub(r'(n\'t)', r' \1',s)
    s = re.sub(r'(\'re|\'s|\'m|\'d)', r' \1',s)
    if not laughters:
        s = re.sub(r'<laughter>', '', s)
    if not disfluencies:
        s = re.sub(r'{\w', '', s)
        s = re.sub(r'}', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s
