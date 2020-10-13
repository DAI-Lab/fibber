import textattack
import torch
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.models.wrappers.model_wrapper import ModelWrapper

from .. import log
from .strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


def tostring(tokenizer, seq):
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(seq))


def compute_clf(clf_model, seq, tok_type):
    return clf_model(seq.unsqueeze(0), token_type_ids=tok_type.unsqueeze(0)
                     )[0].argmax(dim=1)[0].detach().cpu().numpy()


class CLFModel(ModelWrapper):
    def __init__(self, clf_measurement):
        self._clf_measurement = clf_measurement
        self._tokenizer = clf_measurement._tokenizer
        self._context = None

    def __call__(self, text_list):
        res = []
        for item in text_list:
            if self._context is not None:
                res.append(self._clf_measurement.predict_raw(self._context, item))
            else:
                res.append(self._clf_measurement.predict_raw(item, None))
        return res

    def set_context(self, context):
        self._context = context

    def tokenize(self, inputs):
        """Helper method that calls ``tokenizer.batch_encode`` if possible, and
        if not, falls back to calling ``tokenizer.encode`` for each input."""
        if hasattr(self._tokenizer, "batch_encode"):
            return self._tokenizer.batch_encode(inputs)
        else:
            return [self._tokenizer.encode(x) for x in inputs]


class TextFoolerStrategy(StrategyBase):
    def __init__(self, FLAGS, measurement_bundle):
        """Initialize the strategy."""
        super(TextFoolerStrategy, self).__init__(FLAGS, measurement_bundle)

        self._model = CLFModel(measurement_bundle.get_classifier_for_attack())
        self._textfooler = TextFoolerJin2019.build(self._model)

    def paraphrase_example(self, data_record, field_name, n):
        if field_name == "text1":
            self._model.set_context(data_record["text0"])

        att = next(self._textfooler.attack_dataset(
            [(data_record[field_name], data_record["label"])]))
        if isinstance(att, textattack.attack_results.SuccessfulAttackResult):
            return [att.perturbed_result.attacked_text.text]
        else:
            return [data_record[field_name]]
