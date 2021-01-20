def laughter_type(utt):
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

dar_models = {
        'AMI-DA': "AMI-DA-L_bert_2019-11-20",
	'SWDA': "SWDA-L_bert_2019-11-20"
}

