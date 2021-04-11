"""This metric computes the embedding similarity using SBERT model."""

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from fibber import log
from fibber.metrics.metric_base import MetricBase
from fibber.download_utils import get_root_dir

logger = log.setup_custom_logger(__name__)


class SBERTSemanticSimilarityMetric(MetricBase):
    """This metric computes the semantic similarity of two sentences using Sentence BERT model.

    By default the we use stsb-bert-base model.

    see `https://github.com/UKPLab/sentence-transformers` for more information.
    """

    def __init__(self, sbert_pretrained_model="stsb-roberta-large", sbert_gpu_id=-1, **kargs):
        """Initialize sbert model."""
        super(SBERTSemanticSimilarityMetric, self).__init__()

        if sbert_gpu_id == -1:
            logger.warning("SBERT metric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("SBERT metric is running on GPU %d.", sbert_gpu_id)
            self._device = torch.device("cuda:%d" % sbert_gpu_id)

        logger.info("load sbert model.")

        #TODO: use resources utils to manage model.
        sbert_pretrained_model = (get_root_dir() + "/common/transformers_pretrained/"
                                  + sbert_pretrained_model)
        self._model = CrossEncoder(sbert_pretrained_model, device=self._device)

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
        return [float(x) for x in self._model.predict([(origin, paraphrase)
                                                       for paraphrase in paraphrase_list])]

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the perplexity ratio.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.
        """
        return float(self._model.predict([(origin, paraphrase)])[0])
