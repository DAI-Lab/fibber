import re

import numpy as np

from .. import log
from ..resource_utils import get_glove_emb, get_stopwords
from .measurement_base import MeasurementBase

logger = log.setup_custom_logger('glove_semantic_similairty')


def compute_emb(emb_table, id_to_tok, tok_to_id, x):
    x = re.sub(r"[^a-zA-Z0-9]", " ", x)

    toks = x.split()
    embs = []
    for item in toks:
        if item.lower() in tok_to_id:
            embs.append(emb_table[tok_to_id[item.lower()]])
    return np.sum(embs, axis=0)


def compute_emb_sim(emb_table, id_to_tok, tok_to_id, x, y):
    ex = compute_emb(emb_table, id_to_tok, tok_to_id, x)
    ey = compute_emb(emb_table, id_to_tok, tok_to_id, y)
    return ((ex * ey).sum()
            / (np.linalg.norm(ex) + 1e-8)
            / (np.linalg.norm(ey) + 1e-8))


class GloVeSemanticSimilarity(MeasurementBase):
    def __init__(self, **kargs):
        super(GloVeSemanticSimilarity, self).__init__()

        self._glove = get_glove_emb()
        stopwords = get_stopwords()
        logger.info("Glove embeddings and stopwords loaded.")

        for word in stopwords:
            word = word.lower().strip()
            if word in self._glove["tok2id"]:
                self._glove["emb_table"][self._glove["tok2id"][word], :] = 0

    def __call__(self, origin, paraphrase):
        return float(compute_emb_sim(self._glove["emb_table"], self._glove["id2tok"],
                                     self._glove["tok2id"], origin, paraphrase))
