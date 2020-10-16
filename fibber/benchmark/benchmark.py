import argparse
import datetime
import os

from fibber import log
from fibber.benchmark.benchmark_utils import update_detailed_result
from fibber.benchmark.customized_metric_aggregation import customized_metric_aggregation_fn_dict
from fibber.datasets.dataset_utils import get_dataset, subsample_dataset
from fibber.metrics.metric_utils import MetricBundle, aggregate_metrics, compute_metrics
from fibber.paraphrase_strategies.gibbs_sampling_strategy import GibbsSamplingStrategy
from fibber.paraphrase_strategies.gibbs_sampling_wpe_strategy import GibbsSamplingWPEStrategy
from fibber.paraphrase_strategies.gibbs_sampling_wpeb_strategy import GibbsSamplingWPEBStrategy
from fibber.paraphrase_strategies.gibbs_sampling_wpec_strategy import GibbsSamplingWPECStrategy
from fibber.paraphrase_strategies.gibbs_sampling_x_strategy import GibbsSamplingXStrategy
from fibber.paraphrase_strategies.identical_strategy import IdenticalStrategy
from fibber.paraphrase_strategies.random_strategy import RandomStrategy

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ag")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
parser.add_argument("--subsample_testset", type=int, default=1000)
parser.add_argument("--strategy", type=str, default="RandomStrategy")
parser.add_argument("--strategy_gpu", type=int, default=-1)

# metric args
parser.add_argument("--gpt2_gpu", type=int, default=-1)
parser.add_argument("--bert_gpu", type=int, default=-1)
parser.add_argument("--use_gpu", type=int, default=-1)
parser.add_argument("--bert_clf_steps", type=int, default=20000)

RandomStrategy.add_parser_args(parser)
IdenticalStrategy.add_parser_args(parser)

G_EXP_NAME = None


def get_output_filename(FLAGS, prefix="", suffix=""):
    """Returns a string like `<prefix>-<experiment_name>-<suffix>`.

    This function is used to construct filenames for an experiment. This function ensures that
    all filenames for the same experiment has the same experiment name.

    The experiment name is <dataset_name>-<paraphrase_strategy_name>-<date>-<time>.
    """
    global G_EXP_NAME
    if G_EXP_NAME is None:
        G_EXP_NAME = (FLAGS.dataset + "-" + FLAGS.strategy + "-"
                      + datetime.datetime.now().strftime("%m%d-%H%M%S"))
    return prefix + G_EXP_NAME + suffix


def get_strategy(FLAGS, strategy_name, metric_bundle):
    """Take the strategy name and construct a strategy object."""
    if strategy_name == "RandomStrategy":
        return RandomStrategy(FLAGS, metric_bundle)
    if strategy_name == "IdenticalStrategy":
        return IdenticalStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingStrategy":
        return GibbsSamplingStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingXStrategy":
        return GibbsSamplingXStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingWPEStrategy":
        return GibbsSamplingWPEStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingWPEBStrategy":
        return GibbsSamplingWPEBStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingWPECStrategy":
        return GibbsSamplingWPECStrategy(FLAGS, measurement_bundle)
    else:
        assert 0


def benchmark(FLAGS, dataset_name, trainset, testset, paraphrase_set):
    """Run benchmark on the given dataset.

    Args:
        FLAGS (object): args from the argparser.
        dataset_name (str): the name of the dataset.
        trainset (dict): a fibber dataset.
        testset (dict): a fibber dataset.
        paraphrase_set (dict): a fibber dataset.
    """
    logger.info("Build metric bundle.")
    metric_bundle = MetricBundle(
        use_bert_clf_prediction=True,
        use_gpu_id=FLAGS.use_gpu, gpt2_gpu_id=FLAGS.gpt2_gpu,
        bert_gpu_id=FLAGS.bert_gpu, dataset_name=dataset_name,
        trainset=trainset, testset=testset,
        bert_clf_steps=FLAGS.bert_clf_steps)

    paraphrase_strategy = get_strategy(FLAGS, FLAGS.strategy, metric_bundle)
    paraphrase_strategy.fit(trainset)

    tmp_output_filename = os.path.join(
        FLAGS.output_dir, get_output_filename(FLAGS, suffix="-tmp.json"))
    logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
    results = paraphrase_strategy.paraphrase(
        paraphrase_set, FLAGS.num_paraphrases_per_text, tmp_output_filename)

    output_filename = os.path.join(
        FLAGS.output_dir, get_output_filename(FLAGS, suffix="-with-metric.json"))

    results = compute_metrics(metric_bundle, results=results, output_filename=output_filename)

    aggregated_result = aggregate_metrics(
        dataset_name, str(paraphrase_strategy), G_EXP_NAME, results,
        customized_metric_aggregation_fn_dict)
    update_detailed_result(aggregated_result)


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    assert FLAGS.output_dir is not None
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "log"), exist_ok=True)

    log.add_file_handler(
        logger, os.path.join(FLAGS.output_dir, "log", get_output_filename(FLAGS, suffix=".log")))
    log.remove_logger_tf_handler(logger)

    trainset, testset = get_dataset(FLAGS.dataset)
    paraphrase_set = subsample_dataset(testset, FLAGS.subsample_testset)
    logger.info("Subsample test set to %d.", FLAGS.subsample_testset)
    benchmark(FLAGS, FLAGS.dataset, trainset, testset, paraphrase_set)
