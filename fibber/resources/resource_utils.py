import os

import numpy as np
import tqdm

from fibber import log
from fibber.download_utils import download_file, get_root_dir
from fibber.resources.downloadable_resources import downloadable_resource_urls

logger = log.setup_custom_logger(__name__)


def load_glove_model(glove_file, dim):
    """Load glove embeddings from txt file.

    Args:
        glove_file: filename.
        dim: the dimension of the embedding.

    Returns:
        a dictionary:
            "emb_table": a numpy array of size(N, 300)
            "id2tok": a list of strings.
            "tok2id": a dictionary that maps word (string) to its id.
    """
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
        (dict): a dictionary of GloVe word embedding model.
            "emb_table": a numpy array of size(N, 300)
            "id2tok": a list of strings.
            "tok2id": a dictionary that maps word (string) to its id.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common")
    if not os.path.exists(os.path.join(data_dir, "glove.6B.300d.txt")):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls["default-glove-embeddings"])

    return load_glove_model(os.path.join(data_dir, "glove.6B.300d.txt"), 300)


def get_stopwords():
    """Download default stopword words.

    Returns:
        ([str]): a list of strings.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common")
    download_file(subdir=os.path.join(data_dir),
                  **downloadable_resource_urls["default-stopwords"])

    with open(os.path.join(data_dir, "stopwords.txt")) as f:
        stopwords = f.readlines()
    stopwords = [x.strip().lower() for x in stopwords]
    return stopwords
