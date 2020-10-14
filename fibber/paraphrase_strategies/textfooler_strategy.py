"""This module implements the paraphrase strategy using TextFooler."""

import textattack
import torch
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.models.wrappers.model_wrapper import ModelWrapper

from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase


logger = log.setup_custom_logger(__name__)


def tostring(tokenizer, seq):
    """Convert token ids to a string.

    Args:
        tokenizer (object): a Bert tokenizer object.
        seq (list): a list or a numpy array of ints, representing a sequence of word ids.

    Returns:
        (str): the text represented by seq.
    """
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(seq))


def compute_clf(clf_model, seq_tensor, tok_type):
    """Get the prediction label of a sentence.

    Args:
        clf_model (object): a Bert classification model.
        seq_tensor (object): a 1-D int tensor representing the text.
        tok_type (object): a 1-D int tensor with the same length as seq_tensor.
    """
    return clf_model(seq_tensor.unsqueeze(0), token_type_ids=tok_type.unsqueeze(0)
                     )[0].argmax(dim=1)[0].detach().cpu().numpy()


class CLFModel(ModelWrapper):
    """A classifier wrapper for textattack package."""
    def __init__(self, clf_metric):
        self._clf_metric = clf_metric
        self._tokenizer = clf_metric._tokenizer
        self._context = None

    def __call__(self, text_list):
        res = []
        for item in text_list:
            if self._context is not None:
                res.append(self._clf_metric.predict_raw(self._context, item))
            else:
                res.append(self._clf_metric.predict_raw(item, None))
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
    """This strategy uses TextFooler to generate paraphrase.s

    We use TextFooler to attack the classifier in metric_bundle.
        If the attack succeeds, we use the adversarial sentence as a paraphrase.
        If the attack fails, we use the original sentence as a paraphrase.

    This strategy always returns one paraphrase for one data record, regardless of `n`.
    """
    def __init__(self, FLAGS, metric_bundle):
        """Initialize TextFooler."""
        super(TextFoolerStrategy, self).__init__(FLAGS, metric_bundle)

        self._model = CLFModel(metric_bundle.get_classifier_for_attack())
        self._textfooler = TextFoolerJin2019.build(self._model)

    def paraphrase_example(self, data_record, field_name, n):
        """Generate paraphrased sentences."""
        if field_name == "text1":
            self._model.set_context(data_record["text0"])

        att = next(self._textfooler.attack_dataset(
            [(data_record[field_name], data_record["label"])]))
        if isinstance(att, textattack.attack_results.SuccessfulAttackResult):
            return [att.perturbed_result.attacked_text.text]
        else:
            return [data_record[field_name]]
