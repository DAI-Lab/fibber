import argparse
import datetime
import os

from fibber import log
from fibber.benchmark.benchmark_utils import update_detailed_result
from fibber.datasets import builtin_datasets, get_dataset, subsample_dataset, verify_dataset
from fibber.metrics.attack_aggregation_utils import add_sentence_level_adversarial_attack_metrics
from fibber.metrics.metric_utils import MetricBundle
from fibber.paraphrase_strategies import (
    BertSamplingStrategy, IdentityStrategy, RandomStrategy, TextAttackStrategy)
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

built_in_strategies = {
    "RandomStrategy": RandomStrategy,
    "IdentityStrategy": IdentityStrategy,
    "TextAttackStrategy": TextAttackStrategy,
    "BertSamplingStrategy": BertSamplingStrategy
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
                 enable_bert_clf=True,
                 use_gpu_id=-1,
                 gpt2_gpu_id=-1,
                 bert_gpu_id=-1,
                 bert_clf_steps=20000,
                 bert_clf_bs=32):
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
            customized_clf (MetricBase): an classifier object.
            enable_bert_clf (bool): whether to enable bert classifier in metrics. You can disable
                it when you are attacking your own classifier.
            use_gpu_id (int): the gpu to run universal sentence encoder to compute metrics.
                -1 for CPU.
            gpt2_gpu_id (int): the gpu to run the GPT2-medium language model to compute metrics.
                -1 for CPU.
            bert_gpu_id (int): the gpu to run the BERT text classifier, which is the model being
                attacked. -1 for CPU.
            bert_clf_steps (int): number of steps to train the BERT text classifier.
            bert_clf_bs (int): the batch size to train the BERT classifier.
        """
        # make output dir
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._dataset_name = dataset_name

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

        if attack_set is None:
            attack_set = testset

        if subsample_attack_set != 0:
            attack_set = subsample_dataset(attack_set, subsample_attack_set)

        self._trainset = trainset
        self._testset = testset
        self._attack_set = attack_set

        # setup metric bundle
        self._metric_bundle = MetricBundle(
            enable_bert_clf_prediction=enable_bert_clf,
            use_gpu_id=use_gpu_id, gpt2_gpu_id=gpt2_gpu_id,
            bert_gpu_id=bert_gpu_id, dataset_name=dataset_name,
            trainset=self._trainset, testset=testset,
            bert_clf_steps=bert_clf_steps,
            bert_clf_bs=bert_clf_bs
        )

        if customized_clf:
            self._metric_bundle.add_classifier(str(customized_clf), customized_clf)
            self._metric_bundle.set_target_classifier_by_name(str(customized_clf))

        add_sentence_level_adversarial_attack_metrics(
            self._metric_bundle, gpt2_ppl_threshold=5, use_sim_threshold=0.85)

    def run_benchmark(self,
                      paraphrase_strategy="IdentityStrategy",
                      strategy_gpu_id=-1,
                      num_paraphrases_per_text=50,
                      exp_name=None,
                      update_global_results=True):
        """Run the benchmark.

        Args:
            paraphrase_strategy (str or StrategyBase): the paraphrase strategy to benchmark.
                Either the name of a builtin strategy or a customized strategy derived from
                StrategyBase.
            strategy_gpu_id (int): the gpu id to run the strategy. -1 for CPU. Ignored when
                ``paraphrase_strategy`` is an object.
            num_paraphrases_per_text (int): number of paraphrases for each sentence.
            exp_name (str or None): the name of current experiment. None for default name. the
                default name is ``<dataset_name>-<strategy_name>-<date>-<time>``.
            update_global_results (bool): whether to write results in <fibber_root_dir> or the
                benchmark output dir.

        Returns:
            A dict of evaluation results.
        """
        # setup strategy
        if isinstance(paraphrase_strategy, str):
            if paraphrase_strategy in built_in_strategies:
                paraphrase_strategy = built_in_strategies[paraphrase_strategy](
                    {}, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle)
        else:
            assert isinstance(paraphrase_strategy, StrategyBase)

        # get experiment name
        if exp_name is None:
            exp_name = (self._dataset_name + "-" + str(paraphrase_strategy) + "-"
                        + datetime.datetime.now().strftime("%m%d-%H%M%S"))

        paraphrase_strategy.fit(self._trainset)
        tmp_output_filename = os.path.join(
            self._output_dir, exp_name + "-tmp.json")
        logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
        results = paraphrase_strategy.paraphrase_dataset(
            self._attack_set, num_paraphrases_per_text, tmp_output_filename)

        output_filename = os.path.join(
            self._output_dir, exp_name + "-with-metric.json")
        logger.info("Write paraphrase with metrics in %s.", tmp_output_filename)

        results = self._metric_bundle.measure_dataset(
            results=results, output_filename=output_filename)

        aggregated_result = self._metric_bundle.aggregate_metrics(
            self._dataset_name, str(paraphrase_strategy), exp_name, results)
        update_detailed_result(aggregated_result,
                               self._output_dir if not update_global_results else None)
        return aggregated_result

    def get_metric_bundle(self):
        return self._metric_bundle


def get_strategy(arg_dict, dataset_name, strategy_name, strategy_gpu_id,
                 output_dir, metric_bundle):
    """Take the strategy name and construct a strategy object."""
    if strategy_name == "RandomStrategy":
        return RandomStrategy(arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    if strategy_name == "IdentityStrategy":
        return IdentityStrategy(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    if strategy_name == "TextAttackStrategy":
        return TextAttackStrategy(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    if strategy_name == "BertSamplingStrategy":
        return BertSamplingStrategy(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    else:
        assert 0


def main():
    parser = argparse.ArgumentParser()

    # add experiment args
    parser.add_argument("--dataset", type=str, default="ag")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
    parser.add_argument("--subsample_testset", type=int, default=1000)
    parser.add_argument("--strategy", type=str, default="RandomStrategy")
    parser.add_argument("--strategy_gpu_id", type=int, default=-1)

    # metric args
    parser.add_argument("--gpt2_gpu_id", type=int, default=-1)
    parser.add_argument("--bert_gpu_id", type=int, default=-1)
    parser.add_argument("--use_gpu_id", type=int, default=-1)
    parser.add_argument("--bert_clf_steps", type=int, default=20000)

    # add builtin strategies' args to parser.
    for item in built_in_strategies.values():
        item.add_parser_args(parser)

    arg_dict = vars(parser.parse_args())
    assert arg_dict["output_dir"] is not None

    os.makedirs(arg_dict["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(arg_dict["output_dir"], "log"), exist_ok=True)

    benchmark = Benchmark(arg_dict["output_dir"], arg_dict["dataset"],
                          subsample_attack_set=arg_dict["subsample_testset"],
                          use_gpu_id=arg_dict["use_gpu_id"],
                          bert_gpu_id=arg_dict["bert_gpu_id"],
                          gpt2_gpu_id=arg_dict["gpt2_gpu_id"],
                          bert_clf_steps=arg_dict["bert_clf_steps"])

    log.add_file_handler(
        logger, os.path.join(arg_dict["output_dir"], "log.log"))
    log.remove_logger_tf_handler(logger)

    # Get paraphrase strategy
    paraphrase_strategy = get_strategy(arg_dict, arg_dict["dataset"], arg_dict["strategy"],
                                       arg_dict["strategy_gpu_id"], arg_dict["output_dir"],
                                       benchmark.get_metric_bundle())

    benchmark.run_benchmark(paraphrase_strategy=paraphrase_strategy,
                            num_paraphrases_per_text=arg_dict["num_paraphrases_per_text"])


if __name__ == "__main__":
    main()
