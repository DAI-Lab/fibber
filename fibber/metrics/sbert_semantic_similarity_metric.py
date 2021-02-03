"""This metric computes the embedding similarity using SBERT model."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from fibber import log
from fibber.metrics.metric_base import MetricBase

logger = log.setup_custom_logger(__name__)


class SBERTSemanticSimilarityMetric(MetricBase):
    """This metric computes the semantic similarity of two sentences using Sentence BERT model.

    By default the we use stsb-bert-base model.

    see `https://github.com/UKPLab/sentence-transformers` for more information.
    """

    def __init__(self, sbert_pretrained_model="stsb-bert-base", sbert_gpu_id=-1, **kargs):
        """Initialize sbert model."""
        super(SBERTSemanticSimilarityMetric, self).__init__()

        if sbert_gpu_id == -1:
            logger.warning("GPT2 metric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("GPT2 metric is running on GPU %d.", sbert_gpu_id)
            self._device = torch.device("cuda:%d" % sbert_gpu_id)

        logger.info("load sbert model.")

        self._model = SentenceTransformer(sbert_pretrained_model).to(self._device)

    def _get_emb(self, sentences):
        """Compute the embedding of sentences."""
        return self._model.encode(sentences)

    def measure_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0"):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (list): a list containing the USE similarity metric for each paraphrase.
        """
        embs = self._get_emb([origin] + paraphrase_list)
        norm = np.linalg.norm(embs, axis=1)
        sim = np.sum(embs[0] * embs, axis=1) / norm[0] / norm
        return [float(x) for x in sim[1:]]

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the perplexity ratio.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.
        """
        embs = self._get_emb([origin, paraphrase])
        return float(np.sum(embs[0] * embs[1]) / np.linalg.norm(embs[0]) / np.linalg.norm(embs[1]))
