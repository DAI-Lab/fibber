import argparse
import datetime
import os

from fibber import log
from fibber.benchmark.benchmark_utils import update_detailed_result
from fibber.benchmark.customized_metric_aggregation import customized_metric_aggregation_fn_dict
from fibber.datasets import get_dataset, subsample_dataset
from fibber.metrics import MetricBundle, aggregate_metrics, compute_metrics
from fibber.paraphrase_strategies import (
    BertSamplingStrategy, IdentityStrategy, RandomStrategy, TextFoolerStrategy)

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

parser = argparse.ArgumentParser()

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

RandomStrategy.add_parser_args(parser)
IdentityStrategy.add_parser_args(parser)
TextFoolerStrategy.add_parser_args(parser)
BertSamplingStrategy.add_parser_args(parser)

G_EXP_NAME = None


def get_output_filename(arg_dict, prefix="", suffix=""):
    """Returns a string like ``<prefix>-<experiment_name>-<suffix>``.

    This function is used to construct file names for an experiment. This function ensures that
    all file names for the same experiment has the same experiment name.

    The experiment name is ``<dataset_name>-<paraphrase_strategy_name>-<date>-<time>``.
    """
    global G_EXP_NAME
    if G_EXP_NAME is None:
        G_EXP_NAME = (arg_dict["dataset"] + "-" + arg_dict["strategy"] + "-"
                      + datetime.datetime.now().strftime("%m%d-%H%M%S"))
    return prefix + G_EXP_NAME + suffix


def get_strategy(arg_dict, dataset_name, strategy_name, strategy_gpu_id,
                 output_dir, metric_bundle):
    """Take the strategy name and construct a strategy object."""
    if strategy_name == "RandomStrategy":
        return RandomStrategy(arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    if strategy_name == "IdentityStrategy":
        return IdentityStrategy(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    if strategy_name == "TextFoolerStrategy":
        return TextFoolerStrategy(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    if strategy_name == "BertSamplingStrategy":
        return BertSamplingStrategy(
            arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle)
    else:
        assert 0


def benchmark(arg_dict, dataset_name, trainset, testset, paraphrase_set):
    """Run benchmark on the given dataset.

    Args:
        arg_dict (dict): all args from the commandline.
        dataset_name (str): the name of the dataset.
        trainset (dict): a fibber dataset.
        testset (dict): a fibber dataset.
        paraphrase_set (dict): a fibber dataset.
    """
    logger.info("Build metric bundle.")

    metric_bundle = MetricBundle(
        use_bert_clf_prediction=True,
        use_gpu_id=arg_dict["use_gpu_id"], gpt2_gpu_id=arg_dict["gpt2_gpu_id"],
        bert_gpu_id=arg_dict["bert_gpu_id"], dataset_name=dataset_name,
        trainset=trainset, testset=testset,
        bert_clf_steps=arg_dict["bert_clf_steps"])

    paraphrase_strategy = get_strategy(arg_dict, arg_dict["dataset"], arg_dict["strategy"],
                                       arg_dict["strategy_gpu_id"], arg_dict["output_dir"],
                                       metric_bundle)
    paraphrase_strategy.fit(trainset)

    tmp_output_filename = os.path.join(
        arg_dict["output_dir"], get_output_filename(arg_dict, suffix="-tmp.json"))
    logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
    results = paraphrase_strategy.paraphrase_dataset(
        paraphrase_set, arg_dict["num_paraphrases_per_text"], tmp_output_filename)

    output_filename = os.path.join(
        arg_dict["output_dir"], get_output_filename(arg_dict, suffix="-with-metric.json"))

    results = compute_metrics(metric_bundle, results=results, output_filename=output_filename)

    aggregated_result = aggregate_metrics(
        dataset_name, str(paraphrase_strategy), G_EXP_NAME, results,
        customized_metric_aggregation_fn_dict)
    update_detailed_result(aggregated_result)


def main():
    arg_dict = vars(parser.parse_args())
    assert arg_dict["output_dir"] is not None
    os.makedirs(arg_dict["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(arg_dict["output_dir"], "log"), exist_ok=True)

    log.add_file_handler(
        logger, os.path.join(arg_dict["output_dir"], "log",
                             get_output_filename(arg_dict, suffix=".log")))
    log.remove_logger_tf_handler(logger)

    trainset, testset = get_dataset(arg_dict["dataset"])
    paraphrase_set = subsample_dataset(testset, arg_dict["subsample_testset"])
    logger.info("Subsample test set to %d.", arg_dict["subsample_testset"])
    benchmark(arg_dict, arg_dict["dataset"], trainset, testset, paraphrase_set)


if __name__ == "__main__":
    main()
