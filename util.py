import logging
import urllib.request
import shutil
from tqdm import tqdm
from collections import defaultdict

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def create_logger(console_log_level, log_file=None):
    log = logging.getLogger('DARLOGGER')
    log.setLevel(logging.DEBUG)

    # log to the console 
    ch = TqdmLoggingHandler()
    ch.setLevel(console_log_level)
    log.addHandler(ch)

    # log to a file 
    if log_file:
        fh = logging.FileHandler(log_file)
        fh_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

    return log

def rm_dir(directory):
    shutil.rmtree(directory)

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

