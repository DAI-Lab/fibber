"""Top-level package for fibber."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.0.1.dev0'

import os


def get_root_dir():
    """Return `~/.fibber`, the root dir for fibber to store datasets and common resources."""
    root_dir = os.path.join(os.path.expanduser('~'), ".fibber")
    os.makedirs(root_dir, exist_ok=True)
    return root_dir
