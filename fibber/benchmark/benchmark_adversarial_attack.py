import argparse
import datetime
import os

import numpy as np
import torch

from fibber import log
from fibber.benchmark.benchmark_utils import update_attack_robust_result, update_detailed_result
from fibber.datasets import (
    builtin_datasets, clip_sentence, get_dataset, subsample_dataset, verify_dataset)
from fibber.defense_strategies import AdvTrainStrategy, LMAgStrategy, SAPDStrategy, SEMStrategy
from fibber.metrics.attack_aggregation_utils import add_sentence_level_adversarial_attack_metrics
from fibber.metrics.classifier.classifier_base import ClassifierBase
from fibber.metrics.metric_utils import MetricBundle
from fibber.paraphrase_strategies import (
    ASRSStrategy, FudgeStrategy, IdentityStrategy, OpenAttackStrategy, RandomStrategy,
    RemoveStrategy, RewriteRollbackStrategy, SapStrategy, TextAttackStrategy)
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

built_in_paraphrase_strategies = {
    "RandomStrategy": RandomStrategy,
    "IdentityStrategy": IdentityStrategy,
    "TextAttackStrategy": TextAttackStrategy,
    "ASRSStrategy": ASRSStrategy,
    "RewriteRollbackStrategy": RewriteRollbackStrategy,
    "OpenAttackStrategy": OpenAttackStrategy,
    "FudgeStrategy": FudgeStrategy,
    "SapStrategy": SapStrategy,
    "RemoveStrategy": RemoveStrategy,
}

built_in_defense_strategies = {
    "LMAgStrategy": LMAgStrategy,
    "SEMStrategy": SEMStrategy,
    "AdvTrainStrategy": AdvTrainStrategy,
    "SAPDStrategy": SAPDStrategy
}

DATASET_NAME_COL = "0_dataset_name"
STRATEGY_NAME_COL = "1_paraphrase_strategy_name"


class Benchmark(object):
    """Benchmark framework for adversarial attack methods on text classification."""

    def __init__(self,
                 output_dir, dataset_name,
                 trainset=None, testset=None, attack_set=None,
                 subsample_attack_set=0,
                 customized_clf=None,
                 enable_transformer_clf=True,
                 enable_fasttext_classifier=False,
                 use_gpu_id=-1,
                 bert_ppl_gpu_id=-1,
                 transformer_clf_gpu_id=-1,
                 transformer_clf_steps=20000,
                 transformer_clf_bs=32,
                 best_adv_metric_name="USESimilarityMetric",
                 best_adv_metric_lower_better=False,
                 target_classifier="transformer",
                 transformer_clf_model_init="bert-base-cased",
                 field="text0"):
        """Initialize Benchmark framework.

        Args:
            output_dir (str): the directory to write outputs including model, sentences, metrics
                and log.
            dataset_name (str): the name of the dataset.
            trainset (dict): the training set. If the ``dataset_name`` matches built-in datasets,
                ``trainset`` should be None.
            testset (dict): the test set. If the ``dataset_name`` matches built-in datasets,
                ``testset`` should be None.
            attack_set (dict or None): the set to run adversarial attack. Use None to attack the
                ``testset``.
            subsample_attack_set (int): subsample the attack set. 0 to use the whole attack set.
            customized_clf (ClassifierBase): an classifier object.
            enable_transformer_clf (bool): whether to enable transformer classifier in metrics.
            enable_fasttext_classifier (bool): whether to enable fasttext classifier in metrics.
            use_gpu_id (int): the gpu to run universal sentence encoder to compute metrics.
                -1 for CPU.
            bert_ppl_gpu_id (int): the gpu to run the BERT language model for perplexity.
                -1 for CPU.
            transformer_clf_gpu_id (int): the gpu to run the BERT text classifier, which is the
                model being attacked. -1 for CPU.
            transformer_clf_steps (int): number of steps to train the BERT text classifier.
            transformer_clf_bs (int): the batch size to train the BERT classifier.
            best_adv_metric_name (str): the metric name to identify the best adversarial example
                if the paraphrase strategy outputs multiple options.
            best_adv_metric_lower_better (bool): whether the metric is lower better.
            target_classifier (str): the victim classifier. Choose from ["transformer",
                "fasttext", "customized"].
            transformer_clf_model_init (str): the backbone pretrained language model, e.g.,
                `bert-base-cased`.
            field (str): attack text field.
        """
        # make output dir
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._dataset_name = dataset_name
        self._defense_desc = None
        self._field = field

        # verify correct target classifier
        if target_classifier == "customized":
            if not isinstance(customized_clf, ClassifierBase):
                raise RuntimeError("customized_clf is not an instance of ClassifierBase")
        elif target_classifier == "transformer":
            if not enable_transformer_clf:
                raise RuntimeError("target classifier is not enabled.")
        elif target_classifier == "fasttext":
            if not enable_fasttext_classifier:
                raise RuntimeError("target classifier is not enabled.")
        else:
            raise RuntimeError("Unknown target classifier.")

        # setup dataset
        if dataset_name in builtin_datasets:
            if trainset is not None or testset is not None:
                logger.error(("dataset name %d conflict with builtin dataset. "
                              "set trainset and testset to None.") % dataset_name)
                raise RuntimeError
            trainset, testset = get_dataset(dataset_name)
        else:
            verify_dataset(trainset)
            verify_dataset(testset)

        clip_sentence(trainset, transformer_clf_model_init, max_len=128)
        clip_sentence(testset, transformer_clf_model_init, max_len=128)

        if attack_set is None:
            attack_set = testset

        if subsample_attack_set != 0:
            attack_set = subsample_dataset(attack_set, subsample_attack_set)

        self._trainset = trainset
        self._testset = testset
        self._attack_set = attack_set

        # setup metric bundle
        self._metric_bundle = MetricBundle(
            dataset_name=dataset_name,
            trainset=trainset, testset=testset,
            enable_transformer_classifier=enable_transformer_clf,
            transformer_clf_gpu_id=transformer_clf_gpu_id,
            transformer_clf_model_init=transformer_clf_model_init,
            transformer_clf_steps=transformer_clf_steps,
            transformer_clf_bs=transformer_clf_bs,
            enable_fasttext_classifier=enable_fasttext_classifier,
            target_clf=target_classifier,
            enable_use_similarity=True, use_gpu_id=use_gpu_id,
            enable_bert_perplexity=True, bert_ppl_gpu_id=bert_ppl_gpu_id,
            enable_ce_similarity=False,
            enable_gpt2_perplexity=False,
            enable_glove_similarity=False,
            field=field
        )

        if customized_clf:
            self._metric_bundle.add_classifier(customized_clf, True)

        add_sentence_level_adversarial_attack_metrics(
            self._metric_bundle,
            best_adv_metric_name=best_adv_metric_name,
            best_adv_metric_lower_better=best_adv_metric_lower_better)

    def run_benchmark(self,
                      paraphrase_strategy="IdentityStrategy",
                      strategy_gpu_id=-1,
                      max_paraphrases=50,
                      exp_name=None,
                      update_global_results=False):
        """Run the benchmark.

        Args:
            paraphrase_strategy (str or StrategyBase): the paraphrase strategy to benchmark.
                Either the name of a builtin strategy or a customized strategy derived from
                StrategyBase.
            strategy_gpu_id (int): the gpu id to run the strategy. -1 for CPU. Ignored when
                ``paraphrase_strategy`` is an object.
            max_paraphrases (int): number of paraphrases for each sentence.
            exp_name (str or None): the name of current experiment. None for default name. the
                default name is ``<dataset_name>-<strategy_name>-<date>-<time>``.
            update_global_results (bool): whether to write results in <fibber_root_dir> or the
                benchmark output dir.

        Returns:
            A dict of evaluation results.
        """
        # setup strategy
        if isinstance(paraphrase_strategy, str):
            if paraphrase_strategy in built_in_paraphrase_strategies:
                paraphrase_strategy = built_in_paraphrase_strategies[paraphrase_strategy](
                    {}, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle,
                    field=self._field)
        else:
            assert isinstance(paraphrase_strategy, StrategyBase)

        # get experiment name
        if exp_name is None:
            exp_name = (self._dataset_name + "-" + str(paraphrase_strategy) + "-"
                        + datetime.datetime.now().strftime("%m%d-%H%M%S%f"))

        log.add_file_handler(
            logger, os.path.join(self._output_dir, "%s.log" % exp_name))
        log.remove_logger_tf_handler(logger)

        paraphrase_strategy.fit(self._trainset)
        tmp_output_filename = os.path.join(
            self._output_dir, exp_name + "-tmp.json")
        logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
        results = paraphrase_strategy.paraphrase_dataset(
            self._attack_set, max_paraphrases, tmp_output_filename)

        output_filename = os.path.join(
            self._output_dir, exp_name + "-with-metric.json")
        logger.info("Write paraphrase with metrics in %s.", tmp_output_filename)

        results = self._metric_bundle.measure_dataset(
            results=results, output_filename=output_filename)

        aggregated_result = self._metric_bundle.aggregate_metrics(
            self._dataset_name, str(paraphrase_strategy), exp_name, results)

        if self._defense_desc is None:
            update_detailed_result(aggregated_result,
                                   self._output_dir if not update_global_results else None)
        else:
            update_attack_robust_result(aggregated_result,
                                        self._defense_desc,
                                        0,
                                        self._output_dir if not update_global_results else None)

        return aggregated_result

    def get_metric_bundle(self):
        return self._metric_bundle

    def fit_defense(self, paraphrase_strategy, defense_strategy):
        paraphrase_strategy.fit(self._trainset)
        defense_strategy.fit(self._trainset)

    def load_defense(self, defense_strategy):
        clf = defense_strategy.load(self._trainset)
        self._metric_bundle.replace_target_classifier(clf)


def get_strategy(arg_dict, dataset_name, strategy_name, strategy_gpu_id,
                 output_dir, metric_bundle, field):
    """Take the strategy name and construct a strategy object."""
    return built_in_paraphrase_strategies[strategy_name](
        arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle, field)


def get_defense_strategy(arg_dict, dataset_name, strategy_name, strategy_gpu_id,
                         defense_desc, metric_bundle, attack_strategy, field):
    """Take the strategy name and construct a strategy object."""
    return built_in_defense_strategies[strategy_name](
        arg_dict, dataset_name, strategy_gpu_id, defense_desc, metric_bundle,
        attack_strategy, field)


def main():
    parser = argparse.ArgumentParser()

    # target clf
    parser.add_argument("--target_classifier", type=str, default="bert")
    parser.add_argument("--transformer_clf_model_init", type=str, default="bert-base-cased")

    # option on robust defense vs attack
    parser.add_argument("--task", choices=["defense", "attack"], default="attack",
                        help="Choose from defending the classifier vs. attack the classifier.")
    parser.add_argument("--defense_strategy",
                        choices=["none"] + list(built_in_defense_strategies.keys()),
                        default="none", help="use `none` to disable defense. ")
    parser.add_argument("--defense_desc", type=str, default=None,
                        help="A name to recognize this defense setup. ")

    # add experiment args
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="ag")
    parser.add_argument("--field", type=str, default="text0")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_paraphrases", type=int, default=20)
    parser.add_argument("--subsample_testset", type=int, default=1000)
    parser.add_argument("--strategy", choices=list(built_in_paraphrase_strategies.keys()),
                        default="RandomStrategy")
    parser.add_argument("--strategy_gpu_id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # metric args
    parser.add_argument("--bert_ppl_gpu_id", type=int, default=-1)
    parser.add_argument("--transformer_clf_gpu_id", type=int, default=-1)
    parser.add_argument("--use_gpu_id", type=int, default=-1)
    parser.add_argument("--transformer_clf_steps", type=int, default=20000)
    parser.add_argument("--best_adv_metric_name", type=str, default="USESimilarityMetric")
    parser.add_argument("--best_adv_lower_better", type=str, default="0")

    # add builtin strategies' args to parser.
    for item in built_in_paraphrase_strategies.values():
        item.add_parser_args(parser)

    for item in built_in_defense_strategies.values():
        item.add_parser_args(parser)

    arg_dict = vars(parser.parse_args())
    assert arg_dict["output_dir"] is not None

    torch.manual_seed(arg_dict["seed"])
    np.random.seed(arg_dict["seed"])

    benchmark = Benchmark(
        arg_dict["output_dir"], arg_dict["dataset"],
        subsample_attack_set=arg_dict["subsample_testset"],
        use_gpu_id=arg_dict["use_gpu_id"],
        transformer_clf_gpu_id=arg_dict["transformer_clf_gpu_id"],
        bert_ppl_gpu_id=arg_dict["bert_ppl_gpu_id"],
        transformer_clf_steps=arg_dict["transformer_clf_steps"],
        best_adv_metric_name=arg_dict["best_adv_metric_name"],
        best_adv_metric_lower_better=(arg_dict["best_adv_lower_better"] == "1"),
        target_classifier=arg_dict["target_classifier"],
        transformer_clf_model_init=arg_dict["transformer_clf_model_init"],
        field=arg_dict["field"])

    log.remove_logger_tf_handler(logger)

    # Get paraphrase strategy
    paraphrase_strategy = get_strategy(arg_dict, arg_dict["dataset"], arg_dict["strategy"],
                                       arg_dict["strategy_gpu_id"], arg_dict["output_dir"],
                                       benchmark.get_metric_bundle(), field=arg_dict["field"])

    if arg_dict["defense_strategy"] != "none":
        defense_strategy = get_defense_strategy(
            arg_dict, arg_dict["dataset"], arg_dict["defense_strategy"],
            arg_dict["strategy_gpu_id"], arg_dict["defense_desc"], benchmark.get_metric_bundle(),
            attack_strategy=paraphrase_strategy, field=arg_dict["field"])
    else:
        defense_strategy = None

    if arg_dict["task"] == "defense":
        if defense_strategy is None:
            raise RuntimeError("Defense strategy is None.")
        benchmark.fit_defense(paraphrase_strategy=paraphrase_strategy,
                              defense_strategy=defense_strategy)

    elif arg_dict["task"] == "attack":
        if defense_strategy is not None:
            benchmark.load_defense(defense_strategy=defense_strategy)

        benchmark.run_benchmark(paraphrase_strategy=paraphrase_strategy,
                                max_paraphrases=arg_dict["max_paraphrases"],
                                exp_name=arg_dict["exp_name"])
    else:
        raise RuntimeError("Unknown task.")


if __name__ == "__main__":
    main()
