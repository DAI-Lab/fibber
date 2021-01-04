"""This module implements the paraphrase strategy using TextFooler."""

import sys

from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)

try:
    import textattack
    from textattack import attack_recipes
    from textattack.models.wrappers.model_wrapper import ModelWrapper
except ImportError:
    logger.warning("TextAttack is not installed so TextFooler stategy can't be used. "
                   "Install it by `pip install textattack`.")
    ModelWrapper = object


class CLFModel(ModelWrapper):
    """A classifier wrapper for textattack package."""

    def __init__(self, clf_metric, field_name):
        self.model = clf_metric
        self._field_name = field_name
        self._tokenizer = clf_metric._tokenizer
        self._data_record = None

    def __call__(self, text_list):
        ret = self.model.predict_dist_batch(
            self._data_record[self._field_name], text_list,
            data_record=self._data_record, paraphrase_field=self._field_name)
        return ret

    def set_data_record(self, data_record):
        self._data_record = data_record

    def tokenize(self, inputs):
        """Helper method that calls ``tokenizer.batch_encode`` if possible, and
        if not, falls back to calling ``tokenizer.encode`` for each input."""
        if hasattr(self._tokenizer, "batch_encode"):
            return self._tokenizer.batch_encode(inputs)
        else:
            return [self._tokenizer.encode(x) for x in inputs]


class TextAttackStrategy(StrategyBase):
    """This strategy is a wrapper for strategies implemented in TextAttack Package.

    The recipe is used to attack the classifier in metric_bundle.
        If the attack succeeds, we use the adversarial sentence as a paraphrase.
        If the attack fails, we use the original sentence as a paraphrase.

    This strategy always returns one paraphrase for one data record, regardless of `n`.
    """

    __abbr__ = "ta"
    __hyperparameters__ = [
        ("recipe", str, "TextFoolerJin2019", "an attacking recipe implemented in TextAttack.")
    ]

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle):
        if "textattack" not in sys.modules:
            logger.error("TextAttack not installed. Please install textattack manually.")
            raise RuntimeError
        super(TextAttackStrategy, self).__init__(arg_dict, dataset_name, strategy_gpu_id,
                                                 output_dir, metric_bundle)

    def __repr__(self):
        return self._strategy_config["recipe"]

    def fit(self, trainset):
        if ModelWrapper is None:
            raise RuntimeError("no internet connection.")

        self._model = CLFModel(self._metric_bundle.get_target_classifier(),
                               trainset["paraphrase_field"])
        self._recipe = getattr(attack_recipes, self._strategy_config["recipe"]
                               ).build(self._model)

    def paraphrase_example(self, data_record, field_name, n):
        """Generate paraphrased sentences."""
        self._model.set_data_record(data_record)

        attack_text = " ".join(data_record[field_name].split()[:200])
        att = next(self._recipe.attack_dataset(
            [(attack_text, data_record["label"])]))
        if isinstance(att, textattack.attack_results.SuccessfulAttackResult):
            return [att.perturbed_result.attacked_text.text]
        else:
            return [data_record[field_name]]
