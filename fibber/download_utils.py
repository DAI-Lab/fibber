import hashlib
import os
import tarfile

from tensorflow.keras.utils import get_file as tf_get_file

from . import log

logger = log.setup_custom_logger(__name__)


def get_root_dir():
    root_dir = os.path.join(os.path.expanduser('~'), ".fibber")
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def check_file_md5(filename, md5):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == md5


def download_file(filename, url, md5_checksum, subdir=None, untar=False):
    target_dir = get_root_dir()
    if subdir is not None:
        target_dir = os.path.join(target_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    target_file_absolute_path = os.path.join(target_dir, filename)

    if (os.path.exists(target_file_absolute_path) and
            check_file_md5(target_file_absolute_path, md5_checksum)):
        logger.info("Load %s from cache. md5 checksum is correct.", filename)
        if untar:
            my_tar = tarfile.open(target_file_absolute_path)
            my_tar.extractall(target_dir)
            my_tar.close()
    else:
        logger.info("Download %s to %s", filename, target_dir)
        tf_get_file(filename, origin=url, cache_subdir="",
                    file_hash=md5_checksum, extract=untar, cache_dir=target_dir)
