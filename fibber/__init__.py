"""Top-level package for fibber."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.2.dev0'

import os

import nltk


def get_root_dir():
    """Return ``~/.fibber``, the root dir for fibber to store datasets and common resources."""
    root_dir = os.path.join(os.path.expanduser('~'), ".fibber")
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


# change cache directory
nltk.data.path += [os.path.join(get_root_dir(), "common", "nltk_data")]
os.environ['TRANSFORMERS_CACHE'] = os.path.join(
    get_root_dir(), "common", "transformers_pretrained")
os.environ['TFHUB_CACHE_DIR'] = os.path.join(get_root_dir(), "common", "tfhub_pretrained")
os.environ['CORENLP_HOME'] = os.path.join(get_root_dir(), "common", "stanford-corenlp-4.1.0")
