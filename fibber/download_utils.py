"""This module contains utility functions to download files to the root dir ``~/.fibber``."""

import hashlib
import os
import tarfile
import zipfile

from tensorflow.keras.utils import get_file as tf_get_file

from fibber import get_root_dir, log

logger = log.setup_custom_logger(__name__)


def check_file_md5(filename, md5):
    """Check if the md5 of a given file is correct.

    Args:
        filename (str): a filename.
        md5 (str): expected md5 hash value.
    Returns:
        (bool): Return True if md5 matches
    """
    if not os.path.exists(filename):
        return False
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == md5


def download_file(filename, url, md5, subdir=None, untar=False, unzip=False, abs_path=None):
    """Download file from a given url.

    This downloads a file to ``<fibber_root_dir>/subdir``. If the file already exists and the md5
    matches, using the existing file.

    Args:
        filename (str): filename as a string.
        url (str): the url to download the file.
        md5 (str): the md5 checksum of the file.
        subdir (str): the subdir to save the file. Dir will be created if not exists.
        untar (bool): whether to untar the file.
        unzip (bool): whether to unzip the file.
        abs_path (str): a folder to download files. (ignore fibber_root_dir)
    """
    target_dir = get_root_dir()
    if subdir is not None:
        target_dir = os.path.join(target_dir, subdir)
    if abs_path is not None:
        target_dir = abs_path
    os.makedirs(target_dir, exist_ok=True)
    target_file_absolute_path = os.path.join(target_dir, filename)

    if (os.path.exists(target_file_absolute_path)
            and check_file_md5(target_file_absolute_path, md5)):
        logger.info("Load %s from cache. md5 checksum is correct.", filename)
        if untar:
            my_tar = tarfile.open(target_file_absolute_path)
            my_tar.extractall(target_dir)
            my_tar.close()
        if unzip:
            my_zip = zipfile.ZipFile(target_file_absolute_path, "r")
            my_zip.extractall(target_dir)
            my_zip.close()
    else:
        logger.info("Download %s to %s", filename, target_dir)
        tf_get_file(filename, origin=url, cache_subdir="",
                    file_hash=md5, extract=untar or unzip, cache_dir=target_dir)
