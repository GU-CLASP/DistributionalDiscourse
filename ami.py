import xml.etree.ElementTree as ET
import os
from collections import defaultdict, namedtuple
from tqdm import tqdm
import re
import warnings

xml_ns = {"nite": "http://nite.sourceforge.net/"}

inside_parens_re = re.compile(r'\(([^\)]+)\)')

class AMIToken():
    """
    Representation for an element of the AMI trascription words layer.

    Attributes:
        meeting_id  - The ID of the meeting the dialogue is from. 
                       See: http://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml
        speaker      - Which speaker the token is from. One of: {A, B, C, D, E}
        index        - The index of the token in the speaker stream
        token_type   - The tag type in the XML. One of: {w, vocalsound, gap, comment, disfmarker, nonvocalsound,
                     pause, sil, transformerror}
        start_time   - Start timing for the token (for alignment with other speaker streams)
        end_time     - End timing for the token
        punc         - Whether the token is punctutanion (Note: start_time == end_time for punctuation)
        trunc        - Whether the word was truncated
    """

    def __init__(self, meeting_id, speaker, index, token_type, start_time, end_time, text, attrib):
        self.meeting_id = meeting_id
        self.speaker = speaker
        self.index = index
        self.token_type = token_type
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.attrib = attrib

    @classmethod
    def from_xml(cls, element):
        """
        Create an AMIToken from a xml.etree.ElementTree.Element from a words file from the corpus
        """

        meeting_id, speaker, index = cls.__parse_id(element)
        if index is None:
            return None
        token_type = element.tag
        start_time = element.get('starttime')
        end_time   = element.get('endtime')
        if start_time:
            start_time = float(start_time)
        if end_time:
            end_time = float(end_time)
        else:
            end_time = start_time
        text = element.text if element.text else ''
        attrib = element.attrib

        return cls(meeting_id, speaker, index, token_type, start_time, end_time, text, attrib)

    def __str__(self):
        if self.token_type == 'w':
            return self.text
        elif self.token_type == 'vocalsound':
            return "<{}>".format(self.attrib['type'])
        elif self.token_type == 'disfmarker':
            return '<disf>'
        elif self.token_type == 'gap':
            return ''
        elif self.token_type == 'transformerror':  # is this right?
            return '' 
        else:
            raise ValueError("undefined str for token {}".format(self.token_type))

    @staticmethod
    def __parse_id(element):
        index_string = element.get('{{{}}}id'.format(xml_ns['nite']))
        meeting_id, speaker, index = index_string.split('.')
        if index.startswith('wordsx'): # 3 tokens in the corpus have wordsx<id> insteado f words<id>. They apppear to be duplicates
            index = None
        else: 
            index = int(index[5:])
        return meeting_id, speaker, index


class AMIDialogueAct():
    """
    Representation for an element of the AMI da-layer.

    Attributes:
        meeting_id  - The ID of the meeting the dialogue is from. 
                       See: http://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml
        speaker      - Which speaker the DA is from. One of: {A, B, C, D, E}
        index        - The index of the DA in the speaker DA stream
        da-tag      - The DA tag in the AMI scheme
        start_token  - Index of the starting token in the speaker words stream
        end_token    - Index of the end token in the speaker words stream
    """

    def __init__(self, meeting_id, speaker, index, da_tag, start_token, end_token):
        self.meeting_id = meeting_id
        self.speaker = speaker
        self.index = index
        self.da_tag = da_tag
        self.start_token = start_token
        self.end_token = end_token

    @classmethod
    def from_xml(cls, element):

        meeting_id, speaker, _, _, index = element.get('{{{}}}id'.format(xml_ns['nite'])).split('.')

        if len(element) == 2:
            da_tag = inside_parens_re.search(element[0].get('href'))[1]
            token_interval = element[1]
        elif len(element) == 1:
            token_interval = element[0]
            da_tag = None
        token_interval = [i[5:] for i in inside_parens_re.findall(token_interval.get('href'))]

        if len(token_interval) == 2:
            start_token, end_token = token_interval
        else: # len == 1
            start_token = end_token = token_interval[0]
        start_token, end_token = cls.__parse_token_id(start_token), cls.__parse_token_id(end_token)

        return cls(meeting_id, speaker, index, da_tag, start_token, end_token)

    @staticmethod
    def __parse_token_id(id_str):
        return int(id_str.split('.')[-1][5:])

class AMIUtterance():

    def __init__(self, tokens, dialogue_act=None):

        if dialogue_act:
            meeting_id = dialogue_act.meeting_id
            speaker = dialogue_act.speaker
        else:
            meeting_id = tokens[0].meeting_id
            speaker = tokens[0].speaker

        assert(all(t.meeting_id == meeting_id for t in tokens))
        assert(all(t.speaker == speaker for t in tokens))

        self.meeting_id = meeting_id
        self.speaker = speaker
        self.start_time = min(t.start_time for t in tokens)
        self.end_time = max(t.end_time for t in tokens)
        self.dialogue_act = dialogue_act
        self.tokens = tokens

    def text(self):
        return ' '.join(t.text for t in self.tokens)


class AMIMeeting():
    
    def __init__(self, meeting_id, speaker_streams, speaker_dialogue_acts=None):
        
        self.meeting_id = meeting_id
        self.speaker_streams = speaker_streams
        self.speaker_dialogue_acts = speaker_dialogue_acts
        self.speakers = set(speaker_streams.keys())
        self.transcript = None
        
    def gen_transcript(self, split_utts_by_da=True, utt_pause_threshold=1):
        """
        Create a chronological transcript of the dialogue, split by utterances.
        
        Parameters:
            split_utts_by_da - if True, each dialogue act corresponds to a single 
                utterance (assuming dialogue acts are available)
            utt_pause_threshold - if `split_utts_by_da` is False, consecutive utterances 
                by the same speaker are split heuristically by the gap between words.
        """
        if self.transcript:
            return transcript
        
        if split_utts_by_da and self.speaker_dialogue_acts:
            transcript = self.__get_transcript_da()
        else:
            transcript = self.__get_transcript_noda(utt_pause_threshold) 
        
        self.transcript = transcript
        return transcript
    
    def __get_transcript_da(self):
        utts = []
        for speaker in self.speakers:
            dialogue_acts = self.speaker_dialogue_acts[speaker]
            stream = {token.index: token for token in self.speaker_streams[speaker] 
                    if token.start_time is not None and token.end_time is not None}
            for da in dialogue_acts:
                utts.append(AMIUtterance([stream[i] for i in range(da.start_token, da.end_token+1) if i in stream], da))
        utts = sorted(utts, key=lambda x: x.start_time)  # interleave utts
        return utts
    
    def __get_transcript_noda(self, utt_pause_threshold):

        stream = [token for speaker in self.speakers 
                    for token in self.speaker_streams[speaker] 
                    if token.start_time is not None and token.end_time is not None]
        utts = interleave_streams(stream, lambda x: x.speaker, 
                lambda x: x.start_time, lambda x: x.end_time,
                utt_pause_threshold)
        utts = [AMIUtterance(utt) for speaker in self.speakers for utt in utts[speaker]]
        
        return utts

    def get_tokens(self):
        tokens = [token for speaker in self.speaker_streams for token in self.speaker_streams[speaker]
                    if token.start_time]

        tokens = sorted(tokens, key=lambda x:x.start_time)
        return tokens

                
def interleave_streams(stream, speaker, start, end, pause_threshold):
    """
    stream - the interable of objects (e.g., utterances) to be ordered.
    speaker - a function that returtns the speaker for the object in the stream
    start  - a function that returns the start time for objects in the stream
    end    - a function returning the end time for objects in the stream
    """

    utts = defaultdict(list)
    utt_tokens = []
    prev_token = []

    stream = sorted(stream, key=start) 
    while stream:
        token = stream.pop(0)
        if not prev_token:
            utt_tokens.append(token)
        elif (speaker(token) != speaker(prev_token) or start(token) - end(prev_token) > pause_threshold):
            utts[speaker(prev_token)].append(utt_tokens)
            utt_tokens = [token]
        else: 
            utt_tokens.append(token)
        prev_token = token
    utts[speaker(prev_token)].append(utt_tokens)
    return utts

def get_corpus(ami_corpus_dir, da_only=True):

    words_dir = os.path.join(ami_corpus_dir, 'words')
    words_files = [f for f in os.listdir(words_dir) if f.endswith('.xml')]

    da_dir = os.path.join(ami_corpus_dir, 'dialogueActs')
    da_files = [f for f in os.listdir(da_dir) if f.endswith('dialog-act.xml')] # ignore the adjacency-pairs files

    stream_da_acts = defaultdict(lambda: defaultdict(list))
    for file in tqdm(da_files, desc="Reading DA files"):
        tree = ET.parse(os.path.join(da_dir, file))
        root = tree.getroot()
        for element in root:
            da_act = AMIDialogueAct.from_xml(element)
            stream_da_acts[da_act.meeting_id][da_act.speaker].append(da_act)

    if da_only:
        words_files = [w for w in words_files if w.split('.')[0] in stream_da_acts]
    stream_tokens = defaultdict(lambda: defaultdict(list))
    for file in tqdm(words_files, desc="Reading words files"):
        tree = ET.parse(os.path.join(words_dir, file))
        root = tree.getroot()
        for element in root:
            token = AMIToken.from_xml(element)
            if token:
                stream_tokens[token.meeting_id][token.speaker].append(token)

    meetings = [AMIMeeting(meeting_id, stream_tokens[meeting_id], stream_da_acts.get(meeting_id, None)) for meeting_id in stream_tokens ]

    return meetings

if __name__ == '__main__':
    get_corpus('data/AMI/ami_public_manual_1.6.2')
