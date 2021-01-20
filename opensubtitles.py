import re
import glob
import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

laughs = 0


class StreamArray(list):
    """
    https://stackoverflow.com/questions/36157634/how-to-incrementally-write-into-a-json-file

    Converts a generator into a list object that can be json serialisable
    while still retaining the iterative nature of a generator.

    IE. It converts it to a list without having to exhaust the generator
    and keep it's contents in memory.
    """

    def __init__(self, generator):
        self.generator = generator
        self._len = 1

    def __iter__(self):
        self._len = 0
        for item in self.generator:
            yield item
            self._len += 1

    def __len__(self):
        """
        Json parser looks for a this method to confirm whether or not it can
        be parsed
        """
        return self._len


def normalize_laughs(sentence):
    laugh_stems = ["laugh", "chuckl", "giggl", "titter", "scoff"
                   "tee-hee", "snigger", "snicker", "chortl", "guffaw", "roar"]
    regexstr = '(\[|\()\s(\w+\s)*(' + \
        "\w*|".join(laugh_stems) + "\w*)(\s\w+\s)*\s(\]|\))"
    new_sent, lau = re.subn(regexstr, '<laughter>',
                            sentence, flags=re.IGNORECASE)
    global laughs
    laughs += lau
    # if lau:
        # print(new_sent)
    return new_sent, lau


def xml_to_sentences(xml_corpus):
    with open(xml_corpus) as f:
        parse = ET.parse(f)
    sentences = []
    total_laughs = 0
    for sentence in parse.getroot():
        joined_words = " ".join([t.text for t in sentence if t.tag == "w"])
        normalized_words, sentence_laughs = normalize_laughs(joined_words)
        total_laughs += sentence_laughs
        sentences.append(normalized_words)
    return sentences, total_laughs


def os_to_json(path, n_speakers=3, shuffle=False):
    speakers = list(map(chr, range(65, 65+n_speakers)))
    dialogs = []
    # total = len(list(glob.iglob(path + '**/*.xml', recursive=True)))
    # print(f'{total} xml files in the corpus')
    files = glob.iglob(path + '**/*.xml', recursive=True)
    if shuffle:
        files = list(files)
        random.shuffle(files)
    i = 0
    for filename in tqdm(files, total=446612):
        dialog = {}
        dialog['id'] = os.path.basename(filename)
        dialog['utts'], n_laughs = xml_to_sentences(filename)
        dialog['speakers'] = random.choices(speakers, k=len(dialog['utts']))
        yield dialog, n_laughs


if __name__ == '__main__':
    handle = os_to_json(
        '/scratch/DistributionalDiscourse/opensubtitles/OpenSubtitles/xml/en/')
    with open('/scratch/DistributionalDiscourse/opensubtitles/opensubtitles.json', 'w') as f:
        stream_array = StreamArray(handle)
        for chunk in json.JSONEncoder().iterencode(stream_array):
            f.write(chunk)
    print(f'total laughs: {laughs}')

# Local Variables:
# python-shell-interpreter: "nix-shell"
# python-shell-interpreter-args: "--run python"
# End:
