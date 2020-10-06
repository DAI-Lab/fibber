from .bert_clf_flip_pred import BertClfFlipPred
from .editing_distance import EditingDistance
from .glove_semantic_similarity import GloVeSemanticSimilarity
from .gpt2_grammar_quality import GPT2GrammarQuality
from .measurement_base import MeasurementBase
from .use_semantic_similarity import USESemanticSimilarity


class MeasurementBundle(object):
    """docstring for MeasurementBundle."""

    def __init__(self,
                 use_editing_distance=True,
                 use_use_semantic_simialrity=True,
                 use_glove_semantic_similarity=True,
                 use_gpt2_grammar_quality=True,
                 use_bert_clf_flip_pred=False,
                 customized_measurements=[],
                 **kargs):
        super(MeasurementBundle, self).__init__()

        assert isinstance(customized_measurements, list)
        for item in customized_measurements:
            assert isinstance(item, MeasurementBase)

        self._measurements = []
        if use_editing_distance:
            self._measurements.append(EditingDistance(**kargs))
        if use_use_semantic_simialrity:
            self._measurements.append(USESemanticSimilarity(**kargs))
        if use_glove_semantic_similarity:
            self._measurements.append(GloVeSemanticSimilarity(**kargs))
        if use_gpt2_grammar_quality:
            self._measurements.append(GPT2GrammarQuality(**kargs))
        if use_bert_clf_flip_pred:
            self._measurements.append(BertClfFlipPred(**kargs))
        self._measurements += customized_measurements

    def _evaluate(self, origin, paraphrase):
        ret = {}
        for measurement in self._measurements:
            ret[str(measurement)] = measurement(origin, paraphrase)
        return ret

    def __call__(self, origin, paraphrase):
        if isinstance(origin, str):
            assert isinstance(paraphrase, str)
            return self._evaluate(origin, paraphrase)

        assert len(origin) == len(paraphrase)
        ret = []
        for u, v in zip(origin, paraphrase):
            ret.append(self._evaluate(u, v))
        return ret
