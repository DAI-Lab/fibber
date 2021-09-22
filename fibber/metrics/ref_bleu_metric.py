"""This metric computes the embedding similarity using SBERT model."""


from nltk import word_tokenize
from nltk.translate import bleu_score

from fibber import log
from fibber.metrics.metric_base import MetricBase

logger = log.setup_custom_logger(__name__)


class RefBleuMetric(MetricBase):
    """This metric computes the bleu score between input and output"""

    def __init__(self, **kargs):
        """Initialize ce model."""
        super(RefBleuMetric, self).__init__()

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the 4 gram self bleu

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.
        """
        try:
            ref = word_tokenize(data_record["ref"])
        except BaseException:
            logger.warning("Ref not found in data, Ref Blue is set to 0.")
            return 0
        hypo = word_tokenize(paraphrase)
        return bleu_score([ref], hypo)
