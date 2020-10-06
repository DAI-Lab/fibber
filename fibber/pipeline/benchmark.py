import argparse
import datetime
import os

from .. import log
from ..dataset.dataset_utils import get_dataset, subsample_dataset
from ..strategy.random_strategy import RandomStrategy
from .pipeline_utils import measure_quality

logger = log.setup_custom_logger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ag")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
parser.add_argument("--subsample_testset", type=int, default=1000)

parser.add_argument("--measurement_gpu", type=int, default=-1)
parser.add_argument("--strategy_gpu", type=int, default=-1)


G_EXP_NAME = None


def get_output_filename(FLAGS, prefix="", suffix=""):
    global G_EXP_NAME
    if G_EXP_NAME is None:
        G_EXP_NAME = FLAGS.dataset + "-" + datetime.datetime.now().strftime("%m%d-%H%M%S")
    return prefix + G_EXP_NAME + suffix


def benchmark(FLAGS, dataset_name, trainset, testset, paraphrase_set):
    paraphrase_strategy = RandomStrategy()
    paraphrase_strategy.fit(trainset)

    tmp_output_filename = os.path.join(
        FLAGS.output_dir, get_output_filename(FLAGS, suffix="-tmp.json"))
    logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
    results = paraphrase_strategy.paraphrase(
        paraphrase_set, FLAGS.num_paraphrases_per_text, tmp_output_filename)

    output_filename = os.path.join(
        FLAGS.output_dir, get_output_filename(FLAGS, suffix="-with-measurement.json"))
    results = measure_quality(dataset_name=dataset_name, trainset=trainset, testset=testset, results=results,
                              paraphrase_field=paraphrase_set["paraphrase_field"], output_filename=output_filename,
                              measurement_gpu=FLAGS.measurement_gpu)


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    assert FLAGS.output_dir is not None
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "log"), exist_ok=True)

    log.add_filehandler(
        logger, os.path.join(FLAGS.output_dir, "log", get_output_filename(FLAGS, suffix=".log")))

    trainset, testset = get_dataset(FLAGS.dataset)
    paraphrase_set = subsample_dataset(testset, FLAGS.subsample_testset)
    logger.info("Subsample test set to %d.", FLAGS.subsample_testset)
    benchmark(FLAGS, FLAGS.dataset, trainset, testset, paraphrase_set)
