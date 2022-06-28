"""This metric computes the embedding similarity using SBERT model."""


from nltk import word_tokenize
from nltk.translate import bleu_score

from fibber import log
from fibber.metrics.metric_base import MetricBase

logger = log.setup_custom_logger(__name__)


class SelfBleuMetric(MetricBase):
    """This metric computes the bleu score between input and output"""

    def __init__(self, **kwargs):
        """Initialize ce model."""
        super(SelfBleuMetric, self).__init__(**kwargs)

    def _measure_example(self, origin, paraphrase, data_record=None, **kwargs):
        """Compute the 4 gram self bleu

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
        """
        ref = word_tokenize(origin)
        hypo = word_tokenize(paraphrase)
        chencherry = bleu_score.SmoothingFunction()
        return bleu_score.sentence_bleu([ref], hypo, smoothing_function=chencherry.method1)
