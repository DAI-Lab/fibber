"""This metric computes the cosine similarity between two sentences. The sentence embedding is
the sum of GloVe word embeddings."""


import numpy as np
from nltk import word_tokenize

from fibber import log
from fibber.metrics.metric_base import MetricBase
from fibber.resources import get_glove_emb, get_nltk_data, get_stopwords

logger = log.setup_custom_logger('glove_semantic_similairty')


def compute_emb(emb_table, tok_to_id, x):
    """Compute the sum of word embeddings for a sentence.

    Args:
        emb_table (np.array): the glove embedding table.
        tok_to_id (dict): a dict mapping strs to ints.
        x (str): text.

    Returns:
        (np.array): the sum of word embedding.
    """
    toks = word_tokenize(x)
    embs = []
    for item in toks:
        if item.lower() in tok_to_id:
            embs.append(emb_table[tok_to_id[item.lower()]])
    return np.sum(embs, axis=0)


def compute_emb_sim(emb_table, tok_to_id, x, y):
    """Compute the cosine similarity between two sentences. The sentence embedding is the sum
    of word embeddings.

    Args:
        emb_table (np.array): the glove embedding table.
        tok_to_id (dict): a dict mapping strs to ints.
        x (str): text.
        y (str): text.

    Returns:
        (float): the cosine similarity.
    """
    ex = compute_emb(emb_table, tok_to_id, x)
    ey = compute_emb(emb_table, tok_to_id, y)
    return ((ex * ey).sum()
            / (np.linalg.norm(ex) + 1e-8)
            / (np.linalg.norm(ey) + 1e-8))


class GloVeSemanticSimilarityMetric(MetricBase):
    """This metric computes the cosine similarity between two sentences."""

    def __init__(self, **kargs):
        """Initialize, load Glove embeddings."""
        super(GloVeSemanticSimilarityMetric, self).__init__()

        get_nltk_data()
        self._glove = get_glove_emb()
        stopwords = get_stopwords()
        logger.info("Glove embeddings and stopwords loaded.")

        for word in stopwords:
            word = word.lower().strip()
            if word in self._glove["tok2id"]:
                self._glove["emb_table"][self._glove["tok2id"][word], :] = 0

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the Glove cosine similarity between two sentences.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.
        """
        return float(compute_emb_sim(self._glove["emb_table"], self._glove["tok2id"],
                                     origin, paraphrase))
