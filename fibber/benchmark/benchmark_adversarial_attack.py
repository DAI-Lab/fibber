import argparse
import datetime
import os

from fibber import log
from fibber.benchmark.benchmark_utils import update_attack_robust_result, update_detailed_result
from fibber.datasets import (
    builtin_datasets, clip_sentence, get_dataset, subsample_dataset, verify_dataset)
from fibber.metrics.attack_aggregation_utils import add_sentence_level_adversarial_attack_metrics
from fibber.metrics.metric_utils import MetricBundle
from fibber.paraphrase_strategies import (
    ASRSStrategy, FudgeStrategy, IdentityStrategy, OpenAttackStrategy, RandomStrategy,
    RewriteRollbackStrategy, TextAttackStrategy)
from fibber.paraphrase_strategies.strategy_base import StrategyBase
from fibber.robust_tuning_strategy.default_tuning_strategy import (
    DefaultTuningStrategy, TuningStrategyBase)
from fibber.robust_tuning_strategy.naive_tuning_strategy import NaiveTuningStrategy

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
}

built_in_tuning_strategies = {
    "DefaultTuningStrategy": DefaultTuningStrategy,
    "NaiveTuningStrategy": NaiveTuningStrategy
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
                 use_gpu_id=-1,
                 gpt2_gpu_id=-1,
                 bert_ppl_gpu_id=-1,
                 transformer_clf_gpu_id=-1,
                 ce_gpu_id=-1,
                 transformer_clf_steps=20000,
                 transformer_clf_bs=32,
                 load_robust_tuned_clf_desc=None,
                 robust_tuning_steps=0,
                 best_adv_metric_name="USESimilarityMetric",
                 best_adv_metric_lower_better=False,
                 transformer_clf_enable_sem=False,
                 transformer_clf_enable_lmag=False,
                 target_classifier="transformer",
                 transformer_clf_model_init="bert-base"):
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
            enable_transformer_clf (bool): whether to enable bert classifier in metrics. You can disable
                it when you are attacking your own classifier.
            use_gpu_id (int): the gpu to run universal sentence encoder to compute metrics.
                -1 for CPU.
            gpt2_gpu_id (int): the gpu to run the GPT2-medium language model to compute metrics.
                -1 for CPU.
            transformer_clf_gpu_id (int): the gpu to run the BERT text classifier, which is the model being
                attacked. -1 for CPU.
            transformer_clf_steps (int): number of steps to train the BERT text classifier.
            transformer_clf_bs (int): the batch size to train the BERT classifier.
            transformer_clf_enable_sem (bool): use SEM on BERT classifier.
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

        model_init = "bert-base-%s" % ("cased" if trainset["cased"] else "uncased")
        clip_sentence(trainset, model_init, max_len=128)
        clip_sentence(testset, model_init, max_len=128)

        if attack_set is None:
            attack_set = testset

        if subsample_attack_set != 0:
            attack_set = subsample_dataset(attack_set, subsample_attack_set)

        self._trainset = trainset
        self._testset = testset
        self._attack_set = attack_set

        # setup metric bundle
        self._metric_bundle = MetricBundle(
            enable_transformer_classifier=enable_transformer_clf,
            enable_fasttext_classifier=(target_classifier == "fasttext"),
            enable_glove_similarity=False,
            target_clf=target_classifier,
            use_gpu_id=use_gpu_id, gpt2_gpu_id=gpt2_gpu_id,
            transformer_clf_gpu_id=transformer_clf_gpu_id, dataset_name=dataset_name,
            bert_ppl_gpu_id=bert_ppl_gpu_id, enable_bert_perplexity=True,
            trainset=self._trainset, testset=testset,
            transformer_clf_steps=transformer_clf_steps,
            transformer_clf_bs=transformer_clf_bs,
            ce_gpu_id=ce_gpu_id,
            transformer_clf_enable_sem=transformer_clf_enable_sem,
            transformer_clf_enable_lmag=transformer_clf_enable_lmag,
            enable_ce_similarity=False,
            enable_gpt2_perplexity=False,
            transformer_clf_model_init=transformer_clf_model_init
        )

        # self._metric_bundle.get_target_classifier().enable_ppl_filter(
        #     self._metric_bundle.get_metric("BertPerplexityMetric"))

        if customized_clf:
            self._metric_bundle.add_classifier(str(customized_clf), customized_clf)
            self._metric_bundle.set_target_classifier_by_name(str(customized_clf))

        if (load_robust_tuned_clf_desc is not None
                and load_robust_tuned_clf_desc not in ["null", "None", "none", ""]):
            self._metric_bundle.get_target_classifier().load_robust_tuned_model(
                load_robust_tuned_clf_desc, robust_tuning_steps)
            self._robust_tuned_clf_desc = load_robust_tuned_clf_desc
            self._robust_tuning_steps = robust_tuning_steps
        else:
            self._robust_tuned_clf_desc = None
            self._robust_tuning_steps = 0

        add_sentence_level_adversarial_attack_metrics(
            self._metric_bundle,
            best_adv_metric_name=best_adv_metric_name,
            best_adv_metric_lower_better=best_adv_metric_lower_better)

    def run_robust_tuning(self,
                          paraphrase_strategy="IdentityStrategy",
                          tuning_strategy="DefaultTuningStrategy",
                          strategy_gpu_id=-1,
                          num_paraphrases_per_text=50,
                          tuning_steps=5000,
                          num_sentences_to_rewrite_per_step=20):
        """Using a paraphrase strategy to do adversarial fine tuning for the target classifier.

        Args:
            paraphrase_strategy (str or StrategyBase): the paraphrase strategy to benchmark.
                Either the name of a builtin strategy or a customized strategy derived from
                StrategyBase.
            tuning_strategy (str or TuningStrategyBase): the adversarial tuning strategy.
                Either the name of a builtin strategy or a customized strategy derived from
                TuningStrategyBase
            strategy_gpu_id (int): the GPU id to run the paraphrase strategy.
            num_paraphrases_per_text (int): number of paraphrases for each sentence.
            tuning_steps (int): number of steps to tune the classifier.
        """
        if isinstance(paraphrase_strategy, str):
            if paraphrase_strategy in built_in_paraphrase_strategies:
                paraphrase_strategy = built_in_paraphrase_strategies[paraphrase_strategy](
                    {}, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle)
        assert isinstance(paraphrase_strategy, StrategyBase)

        if isinstance(tuning_strategy, str):
            if tuning_strategy in built_in_tuning_strategies:
                robust_tuning_strategy = built_in_tuning_strategies[tuning_strategy]()
        assert isinstance(robust_tuning_strategy, TuningStrategyBase)

        paraphrase_strategy.fit(self._trainset)

        robust_tuning_strategy.fine_tune_classifier(
            metric_bundle=self._metric_bundle,
            paraphrase_strategy=paraphrase_strategy,
            train_set=self._trainset,
            num_paraphrases_per_text=num_paraphrases_per_text,
            tuning_steps=tuning_steps,
            num_sentences_to_rewrite_per_step=num_sentences_to_rewrite_per_step
        )

    def run_benchmark(self,
                      paraphrase_strategy="IdentityStrategy",
                      strategy_gpu_id=-1,
                      num_paraphrases_per_text=50,
                      exp_name=None,
                      update_global_results=False):
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
            if paraphrase_strategy in built_in_paraphrase_strategies:
                paraphrase_strategy = built_in_paraphrase_strategies[paraphrase_strategy](
                    {}, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle)
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
            self._attack_set, num_paraphrases_per_text, tmp_output_filename)

        output_filename = os.path.join(
            self._output_dir, exp_name + "-with-metric.json")
        logger.info("Write paraphrase with metrics in %s.", tmp_output_filename)

        results = self._metric_bundle.measure_dataset(
            results=results, output_filename=output_filename)

        aggregated_result = self._metric_bundle.aggregate_metrics(
            self._dataset_name, str(paraphrase_strategy), exp_name, results)

        if self._robust_tuned_clf_desc is None:
            update_detailed_result(aggregated_result,
                                   self._output_dir if not update_global_results else None)
        else:
            update_attack_robust_result(aggregated_result,
                                        self._robust_tuned_clf_desc,
                                        self._robust_tuning_steps,
                                        self._output_dir if not update_global_results else None)

        return aggregated_result

    def get_metric_bundle(self):
        return self._metric_bundle


def get_strategy(arg_dict, dataset_name, strategy_name, strategy_gpu_id,
                 output_dir, metric_bundle):
    """Take the strategy name and construct a strategy object."""
    return built_in_paraphrase_strategies[strategy_name](
        arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)


def main():
    parser = argparse.ArgumentParser()

    # target clf
    parser.add_argument("--target_classifier", type=str, default="bert")
    parser.add_argument("--transformer_clf_model_init", type=str, default="bert-base")

    # BERT classifier related args
    parser.add_argument("--transformer_clf_enable_sem", type=str, default="0")
    parser.add_argument("--transformer_clf_enable_lmag", type=str, default="0")

    # option on robust training vs attack
    parser.add_argument("--robust_tuning", type=str, default="0",
                        help="use 1 for robust training. (make a separate run to attack).\\"
                             "use 0 for attack.")
    parser.add_argument("--robust_tuning_steps", type=int, default=5000)
    parser.add_argument("--load_robust_tuned_clf_desc", type=str, default=None)

    # add experiment args
    parser.add_argument("--dataset", type=str, default="ag")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
    parser.add_argument("--subsample_testset", type=int, default=1000)
    parser.add_argument("--strategy", type=str, default="RandomStrategy")
    parser.add_argument("--strategy_gpu_id", type=int, default=-1)
    parser.add_argument("--robust_tune_num_attack_per_step", type=int, default=20)

    # metric args
    parser.add_argument("--gpt2_gpu_id", type=int, default=-1)
    parser.add_argument("--bert_ppl_gpu_id", type=int, default=-1)
    parser.add_argument("--transformer_clf_gpu_id", type=int, default=-1)
    parser.add_argument("--use_gpu_id", type=int, default=-1)
    parser.add_argument("--transformer_clf_steps", type=int, default=20000)
    parser.add_argument("--ce_gpu_id", type=int, default=-1)
    parser.add_argument("--best_adv_metric_name", type=str, default="USESimilarityMetric")
    parser.add_argument("--best_adv_lower_better", type=str, default="0")

    # add builtin strategies' args to parser.
    for item in built_in_paraphrase_strategies.values():
        item.add_parser_args(parser)

    arg_dict = vars(parser.parse_args())
    assert arg_dict["output_dir"] is not None

    if arg_dict["robust_tuning"] == "1":
        assert arg_dict["load_robust_tuned_clf_desc"] is None

    benchmark = Benchmark(
        arg_dict["output_dir"], arg_dict["dataset"],
        subsample_attack_set=arg_dict["subsample_testset"],
        use_gpu_id=arg_dict["use_gpu_id"],
        transformer_clf_gpu_id=arg_dict["transformer_clf_gpu_id"],
        gpt2_gpu_id=arg_dict["gpt2_gpu_id"],
        bert_ppl_gpu_id=arg_dict["bert_ppl_gpu_id"],
        transformer_clf_steps=arg_dict["transformer_clf_steps"],
        load_robust_tuned_clf_desc=arg_dict["load_robust_tuned_clf_desc"],
        robust_tuning_steps=arg_dict["robust_tuning_steps"],
        ce_gpu_id=arg_dict["ce_gpu_id"],
        best_adv_metric_name=arg_dict["best_adv_metric_name"],
        best_adv_metric_lower_better=(arg_dict["best_adv_lower_better"] == "1"),
        transformer_clf_enable_sem=(arg_dict["transformer_clf_enable_sem"] == "1"),
        transformer_clf_enable_lmag=(arg_dict["transformer_clf_enable_lmag"] == "1"),
        target_classifier=arg_dict["target_classifier"],
        transformer_clf_model_init=arg_dict["transformer_clf_model_init"])

    log.remove_logger_tf_handler(logger)

    # Get paraphrase strategy
    paraphrase_strategy = get_strategy(arg_dict, arg_dict["dataset"], arg_dict["strategy"],
                                       arg_dict["strategy_gpu_id"], arg_dict["output_dir"],
                                       benchmark.get_metric_bundle())

    if arg_dict["robust_tuning"] == "1":
        benchmark.run_robust_tuning(paraphrase_strategy=paraphrase_strategy,
                                    num_paraphrases_per_text=arg_dict["num_paraphrases_per_text"],
                                    tuning_strategy="DefaultTuningStrategy",
                                    tuning_steps=arg_dict["robust_tuning_steps"],
                                    num_sentences_to_rewrite_per_step=arg_dict[
                                        "robust_tune_num_attack_per_step"])
    else:
        benchmark.run_benchmark(paraphrase_strategy=paraphrase_strategy,
                                num_paraphrases_per_text=arg_dict["num_paraphrases_per_text"])


if __name__ == "__main__":
    main()
