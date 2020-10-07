import os

import numpy as np
import tqdm

from . import log
from .download_utils import download_file, get_root_dir
from .downloadable_resources import resources

logger = log.setup_custom_logger(__name__)


def load_glove_model(glove_file, dim):
    glove_file_lines = open(glove_file, 'r').readlines()

    emb_table = np.zeros((len(glove_file_lines), dim), dtype='float32')
    id_to_tok = []
    tok_to_id = {}

    logger.info("Load glove embeddings as np array.")
    id = 0
    for line in tqdm.tqdm(glove_file_lines):
        split_line = line.split()
        word = split_line[0]
        emb_table[id] = np.array([float(val) for val in split_line[1:]])
        id_to_tok.append(word)
        tok_to_id[word] = id
        id += 1

    return {
        "emb_table": emb_table,
        "id2tok": id_to_tok,
        "tok2id": tok_to_id
    }


def get_glove_emb():
    """Download default pretrained glove embeddings and return a dict.

    We use the 300-dimensional model trained on Wikipedia 2014 + Gigaword 5.
    See https://nlp.stanford.edu/projects/glove/

    Returns:
        a dictionary:
            "emb_table": a numpy array of size(N, 300)
            "id2tok": a list of strings.
            "tok2id": a dictionary that maps word (string) to its id.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common")
    download_file(filename=resources["default-glove-embeddings"]["filename"],
                  url=resources["default-glove-embeddings"]["url"],
                  md5_checksum=resources["default-glove-embeddings"]["md5"],
                  subdir=os.path.join(data_dir), untar=True)

    return load_glove_model(os.path.join(data_dir, "glove.6B.300d.txt"), 300)


def get_stopwords():
    """Download default stopword words.

    Returns:
        a list of strings.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common")
    download_file(filename=resources["default-stopwords"]["filename"],
                  url=resources["default-stopwords"]["url"],
                  md5_checksum=resources["default-stopwords"]["md5"],
                  subdir=os.path.join(data_dir), untar=False)

    with open(os.path.join(data_dir, "stopwords.txt")) as f:
        stopwords = f.readlines()
    stopwords = [x.strip().lower() for x in stopwords]
    return stopwords


def get_dataset_result_filename(dataset_name):
    result_dir = os.path.join(get_root_dir(), "results")
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, "%s.csv" % dataset_name)
