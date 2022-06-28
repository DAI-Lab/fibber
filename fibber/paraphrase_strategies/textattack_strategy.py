"""This module implements the paraphrase strategy using TextFooler."""

import signal
import sys
import traceback

from nltk import word_tokenize

from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)

try:
    sys.path.append("./TextAttack")
    import textattack
    from textattack import attack_recipes
    from textattack.models.wrappers.model_wrapper import ModelWrapper
except ImportError:
    logger.warning("TextAttack is not installed so TextFooler stategy can't be used. "
                   "Install it by `pip install textattack`.")
    ModelWrapper = object


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()


class DefaultTokenizer(object):
    def batch_encode(self, texts):
        return [self.encode(item) for item in texts]

    def encode(self, text):
        return word_tokenize(text)


class CLFModel(ModelWrapper):
    """A classifier wrapper for textattack package."""

    def __init__(self, clf_metric, field):
        self.model = clf_metric
        self._field = field
        if hasattr(clf_metric, "_tokenizer"):
            self._tokenizer = clf_metric._tokenizer
        else:
            self._tokenizer = DefaultTokenizer()
        self._data_record = None
        self._counter = 0

    def __call__(self, text_list):
        self._counter += len(text_list)
        ret = self.model.predict_log_dist_batch(
            self._data_record[self._field], text_list, data_record=self._data_record)

        return ret

    def reset_counter(self):
        self._counter = 0

    def get_counter(self):
        return self._counter

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
        ("recipe", str, "TextFoolerJin2019", "an attacking recipe implemented in TextAttack."),
        ("time_limit", float, 60, "time limit for each attack.")
    ]

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle, field):
        if "textattack" not in sys.modules:
            logger.error("TextAttack not installed. Please install textattack manually.")
            raise RuntimeError
        super(TextAttackStrategy, self).__init__(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle, field)

    def __repr__(self):
        return self._strategy_config["recipe"]

    def fit(self, trainset):
        if ModelWrapper is None:
            raise RuntimeError("no internet connection.")

        self._model = CLFModel(self._metric_bundle.get_target_classifier(), self._field)
        self._recipe = getattr(attack_recipes, self._strategy_config["recipe"]
                               ).build(self._model)

    def paraphrase_example(self, data_record, n):
        """Generate paraphrased sentences."""
        self._model.set_data_record(data_record)

        self._model.reset_counter()

        attack_text = data_record[self._field]

        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(self._strategy_config["time_limit"])
        try:
            att = self._recipe.attack(attack_text, data_record["label"])
        except TimeOutException:
            logger.warn("Timeout.")
            att = None
        except RuntimeError:
            logger.warn("TextAttack package failure.")
            traceback.print_exc()
            att = None
        signal.alarm(0)

        clf_count = self._model.get_counter()
        if isinstance(att, textattack.attack_results.SuccessfulAttackResult):
            return [att.perturbed_result.attacked_text.text], clf_count
        else:
            return [data_record[self._field]], clf_count
