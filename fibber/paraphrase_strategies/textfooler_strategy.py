"""This module implements the paraphrase strategy using TextFooler."""


from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)

try:
    import textattack
    from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
    from textattack.models.wrappers.model_wrapper import ModelWrapper
except ImportError:
    logger.warning("TextAttack is not installed so TextFooler stategy can't be used. "
                   "Install it by `pip install textattack`.")
    ModelWrapper = object


class CLFModel(ModelWrapper):
    """A classifier wrapper for textattack package."""

    def __init__(self, clf_metric):
        self._clf_metric = clf_metric
        self._tokenizer = clf_metric._tokenizer
        self._context = None

    def __call__(self, text_list):
        ret = self._clf_metric.predict_dist_batch(text_list, self._context)
        return ret

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

    __abbr__ = "tf"

    def fit(self, trainset):
        if ModelWrapper is None:
            raise RuntimeError("no internet connection.")

        self._model = CLFModel(self._metric_bundle.get_classifier_for_attack())
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
