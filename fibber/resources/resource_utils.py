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
        a dict:
            "emb_table": a numpy array of size(N, 300)
            "id2tok": a list of strings.
            "tok2id": a dict that maps word (string) to its id.
    """
    glove_file_lines = open(glove_file, "r", encoding="utf8").readlines()

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


def get_glove_emb(download_only=False):
    """Download default pretrained glove embeddings and return a dict.

    We use the 300-dimensional model trained on Wikipedia 2014 + Gigaword 5.
    See https://nlp.stanford.edu/projects/glove/

    Args:
        download_only (bool): set True to only download. (Returns None)

    Returns:
        (dict): a dict of GloVe word embedding model.
            "emb_table": a numpy array of size(N, 300)
            "id2tok": a list of strings.
            "tok2id": a dict that maps word (string) to its id.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common")
    if not os.path.exists(os.path.join(data_dir, "glove.6B.300d.txt")):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls["default-glove-embeddings"])

    if download_only:
        return None
    return load_glove_model(os.path.join(data_dir, "glove.6B.300d.txt"), 300)


def get_counter_fitted_vector(download_only=False):
    """Download default pretrained counter fitted embeddings and return a dict.

    See https://github.com/nmrksic/counter-fitting

    Args:
        download_only (bool): set True to only download. (Returns None)

    Returns:
        (dict): a dict of GloVe word embedding model.
            "emb_table": a numpy array of size(N, 300)
            "id2tok": a list of strings.
            "tok2id": a dict that maps word (string) to its id.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common")
    if not os.path.exists(os.path.join(data_dir, "counter-fitted-vectors.txt")):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls["counter-fitted-vectors"])

    if download_only:
        return None
    return load_glove_model(os.path.join(data_dir, "counter-fitted-vectors.txt"), 300)


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


def get_nltk_data():
    """Download nltk data to ``<fibber_root_dir>/nltk_data``."""
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common", "nltk_data", "tokenizers")
    if not os.path.exists(os.path.join(data_dir, "punkt")):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls["nltk-punkt"])

    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common", "nltk_data", "corpora")
    if not os.path.exists(os.path.join(data_dir, "stopwords")):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls["nltk_stopwords"])


def get_universal_sentence_encoder():
    """Download pretrained universal sentence encoder.

    Returns:
        (str): directory of the downloaded model.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common", "tfhub_pretrained",
                            "universal-sentence-encoder-large_5")
    if not os.path.exists(data_dir):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls["universal-sentence-encoder"])

    return data_dir


def get_transformers(name):
    """Download pretrained transformer models.

    Args:
        name (str): the name of the pretrained models. options are ``["bert-base-cased",
            "bert-base-uncased", "gpt2-medium"]``.

    Returns:
        (str): directory of the downloaded model.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "common", "transformers_pretrained")
    if not os.path.exists(os.path.join(data_dir, name)):
        download_file(subdir=os.path.join(data_dir),
                      **downloadable_resource_urls[name])

    return os.path.join(data_dir, name)


def get_bert_clf_demo():
    """Download the pretrained classifier for demo dataset."""
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "transformer_clf")
    if not os.path.exists(os.path.join(data_dir, "demo")):
        download_file(subdir=data_dir,
                      **downloadable_resource_urls["bert-base-cased-clf-demo"])


def get_bert_lm_demo():
    """Download the pretrained language model for demo dataset."""
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "bert_lm")
    if not os.path.exists(os.path.join(data_dir, "demo")):
        download_file(subdir=data_dir,
                      **downloadable_resource_urls["bert-base-cased-lm-demo"])


def get_wordpiece_emb_demo():
    """Download wordpiece embeddings for demo dataset."""
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "wordpiece_emb_conterfited")
    if not os.path.exists(os.path.join(data_dir, "wordpiece_emb-demo-0500.pt")):
        download_file(subdir=data_dir, **downloadable_resource_urls["wpe-demo"])
