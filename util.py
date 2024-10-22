import logging
import urllib.request
import shutil
from tqdm import tqdm
from collections import defaultdict
import os

def pad_lists(ls, min_len=None, max_len=None, pad=0):
    pad_len = max(len(l) for l in ls)
    if max_len:
        pad_len = min(pad_len, max_len)
    if min_len:
        pad_len = max(pad_len, min_len)
    return [(l + ([pad] * (pad_len - len(l))))[-pad_len:] for l in ls]

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

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def rm_dir(directory):
    shutil.rmtree(directory)

