import argparse
import logging
import json
import zipfile
import tarfile
import os

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

DATA_DIR = 'data'


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


class Utterance():

    def __init__(self, speaker, da_tag, text):
        self.speaker = speaker
        self.da_tag = da_tag
        self.text = text

    def to_dict(self):
        return {'speaker': self.speaker,
                'da_tag': self.da_tag,
                'text': f"[SPKR_{self.speaker}] {self.text}"}


class Dialogue():

    def __init__(self, dialogue_id, utts):
        self.id = dialogue_id
        self.speakers = list({utt.speaker for utt in utts})
        self.utts = utts

    def to_dict(self):
        return {
            'id': self.id,
            'speakers': self.speakers,
            'utts': [utt.to_dict() for utt in self.utts]}


class DialogueCorpus():
    """
    Standardized corpus format for dialogue act recoginiton.
    """

    def __init__(self, corpus_name, corpus_dir, corpus_file):
        self.name = corpus_name
        self.corpus_dir = os.path.join(DATA_DIR, corpus_dir)
        self.corpus_file = os.path.join(DATA_DIR, corpus_file)

    def to_json(self):
        with open(self.corpus_file, 'w') as f:
            dialogues = [dialogue.to_dict() for dialogue in self.dialogues]
            json.dump(dialogues, f)
        log.info(f"Wrote {self.name} ({len(dialogues)} dialogues) to {self.corpus_file}.")

    def download_corpus():
        raise NotImplementedError

    def parse_corpus(self):
        raise NotImplementedError


class AMICorpus(DialogueCorpus):

    def __init__(self, da_only=True):
        """
        da_only - whether to restrict the corpus to DA-tagged dialogues
        """
        corpus_name = "AMI Meeting Corpus"
        corpus_dir = "AMI"
        corpus_file = "AMI-DA.json" if da_only else "AMI.json"
        if da_only:
            corpus_name += " (dialogue act-tagged)"
        self.da_only = da_only
        super().__init__(corpus_name, corpus_dir, corpus_file)

    def download_corpus(self):
        url = "http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        zipfilename = os.path.join(DATA_DIR, "ami_public_manual_1.6.2.zip")
        util.download_url(url, zipfilename)
        with zipfile.ZipFile(zipfilename, 'r') as f:
            f.extractall(self.corpus_dir)
        os.remove(zipfilename)

    def parse_corpus(self):
        ami_meetings = ami.get_corpus(self.corpus_dir, self.da_only)
        dialogues = []
        for m in ami_meetings:
            m.gen_transcript(split_utts_by_da=self.da_only)
            utts = [Utterance(u.speaker, u.dialogue_act.da_tag, u.text()) 
                    for u in m.transcript]
            dialogues.append(Dialogue(m.meeting_id, utts))
        self.dialogues = dialogues


class SWBDWordAlignedCorpus(DialogueCorpus):

    def __init__(self):
        corpus_name = "Switchboard Corpus"
        corpus_dir = "SWBD"
        corpus_file = "SWBD.json"
        super().__init__(corpus_name, corpus_dir, corpus_file)

    def download_corpus(self):
        url = "http://www.isip.piconepress.com/projects/switchboard/releases/ptree_word_alignments.tar.gz"
        zipfilename = os.path.join(DATA_DIR, "ptree_word_alignments.tar.gz")
        util.download_url(url, zipfilename)
        with tarfile.open(zipfilename, "r:gz") as f:
            f.extractall(self.corpus_dir)
        os.remove(zipfilename)


class SWDACorpus(DialogueCorpus):

    def __init__(self):
        corpus_name = "Switchboard Dialogue Act Corpus"
        corpus_dir = "SWDA"
        corpus_file = "SWDA.json"
        super().__init__(corpus_name, corpus_dir, corpus_file)

    def download_corpus(self):
        url = "https://github.com/cgpotts/swda/blob/master/swda.zip?raw=true"
        zipfilename = os.path.join(DATA_DIR, "swda.zip")
        util.download_url(url, zipfilename)
        with zipfile.ZipFile(zipfilename, 'r') as f:
            f.extractall(self.corpus_dir)
        os.remove(zipfilename)

    def parse_corpus(self):
        corpus = swda.CorpusReader(os.path.join(self.corpus_dir,'swda'))
        dialogues = []
        for transcript in corpus.iter_transcripts():
           utts = [Utterance(u.caller, damsl_tag_cluster(u.act_tag), self.normalize(u.text.lower()))
                   for u in transcript.utterances]
           dialogues.append(Dialogue(transcript.conversation_no, utts))
        self.dialogues = dialogues

    def normalize(self, text):
        return tokenize(text)


if __name__ == '__main__':

    args = parser.parse_args()
    log = util.create_logger(logging.INFO)

    corpora = {
        'swda': SWDACorpus(),
        'swbd': SWBDWordAlignedCorpus(),
        'ami':  AMICorpus(da_only=False),
        'ami-da': AMICorpus(da_only=True)}

    if args.command == 'prep-corpora':

        if not args.corpora:
            corpora = list(corpora)
        else:
            corpora = [corpora[c] for c in corpora if c in args.corpora]

        for corpus in corpora:
            if os.path.exists(corpus.corpus_dir):
                log.info(f"{corpus.corpus_dir} already exists." 
                         "Delete this directory if you want to re-download the corpus.")
            else:
                corpus.download_corpus()
            if os.path.isfile(corpus.corpus_file):
                log.info(f"{corpus.corpus_file} already exists."
                        "Delete this file if you want to preprocess the corpus again.")
            else:
                corpus.parse_corpus()
                corpus.to_json()

    if args.command == 'customize-bert-vocab':
        pass

