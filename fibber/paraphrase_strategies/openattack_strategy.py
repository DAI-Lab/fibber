"""This module implements the paraphrase strategy using TextFooler."""

import numpy as np
import OpenAttack as oa

from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


class MyClassifier(oa.Classifier):
    def __init__(self, clf_metric, field_name):
        self.model = clf_metric
        self._field_name = field_name
        self._data_record = None

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def set_data_record(self, data_record):
        self._data_record = data_record.copy()

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = self.model.predict_dist_batch(
            self._data_record[self._field_name], input_,
            data_record=self._data_record, paraphrase_field=self._field_name)
        return np.exp(ret)


class OpenAttackStrategy(StrategyBase):
    """This strategy is a wrapper for strategies implemented in OpenAttack Package.

    The recipe is used to attack the classifier in metric_bundle.
        If the attack succeeds, we use the adversarial sentence as a paraphrase.
        If the attack fails, we use the original sentence as a paraphrase.

    This strategy always returns one paraphrase for one data record, regardless of `n`.
    """

    __abbr__ = "oa"
    __hyperparameters__ = [
        ("recipe", str, "PSOAttacker", "an attacking recipe implemented in OpenAttack."),
    ]

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle):
        super(OpenAttackStrategy, self).__init__(arg_dict, dataset_name, strategy_gpu_id,
                                                 output_dir, metric_bundle)

    def __repr__(self):
        return self._strategy_config["recipe"]

    def fit(self, trainset):
        self._victim = MyClassifier(self._metric_bundle.get_target_classifier(),
                                    trainset["paraphrase_field"])
        self._attacker = getattr(oa.attackers, self._strategy_config["recipe"])()

    def paraphrase_example(self, data_record, field_name, n):
        """Generate paraphrased sentences."""
        self._victim.set_data_record(data_record)

        attack_text = " ".join(data_record[field_name])
        attack_eval = oa.AttackEval(self._attacker, self._victim)
        res = next(attack_eval.ieval([{"x": attack_text, "y": data_record["label"]}]))
        return [res["result"]] if res["success"] else [attack_text]
