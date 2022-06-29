import copy

import numpy as np
import tqdm

from fibber import log
from fibber.datasets import subsample_dataset
from fibber.defense_strategies.defense_strategy_base import DefenseStrategyBase

logger = log.setup_custom_logger(__name__)


class AdvTrainStrategy(DefenseStrategyBase):
    """Base class for Tuning strategy"""

    __abbr__ = "adv"
    __hyperparameters__ = [
        ("epoch", int, 5, "number of rewrites."),
        ("sample", int, 5000, "number of training set to subsample."),
        ("bs", int, 32, "classifier training batch size."),
        ("alpha", float, 0.2, "the ratio of adversarial examples")]

    def fit(self, trainset):
        n_epoch = self._strategy_config["epoch"]
        bs = self._strategy_config["bs"]
        sample = self._strategy_config["sample"]
        alpha = self._strategy_config["alpha"]

        steps_est = int(n_epoch * min(sample, len(trainset["data"])) * (1 + alpha) / bs)
        self._classifier.robust_tune_init(optimizer="adamw", lr=1e-5, weight_decay=0.001,
                                          steps=steps_est)

        for i in range(n_epoch):
            trainset_sampled = subsample_dataset(trainset, sample)
            np.random.shuffle(trainset_sampled["data"])

            logger.info("Constructing adversarial examples.")
            adv_list = []
            pbar = tqdm.tqdm(total=len(trainset_sampled["data"]))
            for item in trainset_sampled["data"]:
                paraphrase_list, _ = self._attack_strategy.paraphrase_example(item, 16)
                pred_list = self._classifier.measure_batch(
                    item[self._field], paraphrase_list, item)
                for sent, pred in zip(paraphrase_list, pred_list):
                    if pred != item["label"]:
                        tmp = copy.deepcopy(item)
                        tmp[self._field] = sent
                        adv_list.append(tmp)
                pbar.update(1)
                pbar.set_postfix({"advs": len(adv_list)})
                if len(adv_list) >= alpha * len(trainset_sampled["data"]):
                    break
            trainset_sampled["data"] += adv_list

            logger.info("Training.")
            np.random.shuffle(trainset_sampled["data"])
            for j in range(0, len(trainset_sampled["data"]), bs):
                self._classifier.robust_tune_step(trainset_sampled["data"][j:j + bs])

        self._classifier.save_robust_tuned_model(self._defense_save_path)

    def load(self, trainset):
        self._classifier.load_robust_tuned_model(self._defense_save_path)
        return self.__classifier
