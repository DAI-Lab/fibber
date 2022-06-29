"""This metric computes the embedding similarity using SBERT model."""

from sentence_transformers import CrossEncoder

from fibber import log, resources
from fibber.metrics.metric_base import MetricBase

logger = log.setup_custom_logger(__name__)


class CESimilarityMetric(MetricBase):
    """This metric computes the semantic similarity of two sentences using Cross Encoder model.

    By default the we use stsb-roberta-large model.

    see `https://github.com/UKPLab/sentence-transformers` for more information.
    """

    def __init__(self, ce_pretrained_model="stsb-roberta-large", ce_gpu_id=-1, **kwargs):
        """Initialize ce model."""
        super(CESimilarityMetric, self).__init__(**kwargs)

        if ce_gpu_id == -1:
            logger.warning("CE metric is running on CPU.")
            device = "cpu"
        else:
            logger.info("CE metric is running on GPU %d.", ce_gpu_id)
            device = "cuda:%d" % ce_gpu_id

        logger.info("load ce model.")

        self._model = CrossEncoder(resources.get_transformers(ce_pretrained_model),
                                   device=device)

    def _get_emb(self, sentences):
        """Compute the embedding of sentences."""
        return self._model.encode(sentences)

    def _measure_batch(self, origin, paraphrase_list, data_record=None, **kwargs):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.

        Returns:
            (list): a list containing the USE similarity metric for each paraphrase.
        """
        return [float(x) for x in self._model.predict(
            [[origin, paraphrase] for paraphrase in paraphrase_list])]

    def _measure_multiple_examples(self, origin_list, paraphrase_list,
                                   data_record_list=None, **kwargs):
        assert len(origin_list) == len(paraphrase_list)
        return [float(x) for x in self._model.predict(
            [[origin, paraphrase] for origin, paraphrase in zip(origin_list, paraphrase_list)])]

    def _measure_example(self, origin, paraphrase, data_record=None, **kwargs):
        """Compute the perplexity ratio.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
        """
        return float(self._model.predict([[origin, paraphrase]])[0])
