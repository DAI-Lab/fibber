import argparse
import datetime
import os

from .. import log
from ..dataset.dataset_utils import get_dataset, subsample_dataset
from ..measurement.measurement_utils import aggregate_measurements, measure_quality
from ..resource_utils import get_dataset_result_filename
from ..strategy.random_strategy import RandomStrategy

logger = log.setup_custom_logger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ag")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
parser.add_argument("--subsample_testset", type=int, default=1000)

parser.add_argument("--strategy", type=str, default="RandomStrategy")

parser.add_argument("--gpt2_gpu", type=int, default=-1)
parser.add_argument("--bert_gpu", type=int, default=-1)
parser.add_argument("--use_gpu", type=int, default=-1)

RandomStrategy.add_parser_args(parser)

G_EXP_NAME = None


def get_output_filename(FLAGS, prefix="", suffix=""):
    global G_EXP_NAME
    if G_EXP_NAME is None:
        G_EXP_NAME = (FLAGS.dataset + "-" + FLAGS.strategy + "-"
                      + datetime.datetime.now().strftime("%m%d-%H%M%S"))
    return prefix + G_EXP_NAME + suffix


def flip_pred_agg_fn(use_sim, ppl_score):
    def agg_fn(paraphrase_measurement):
        for item in paraphrase_measurement:
            if (item["BertClfFlipPred"] == 1
                and item["GPT2GrammarQuality"] < ppl_score
                    and item["USESemanticSimilarity"] < use_sim):
                return 1
        return 0
    return agg_fn


def get_strategy(FLAGS, strategy_name):
    if strategy_name == "RandomStrategy":
        return RandomStrategy(FLAGS)
    else:
        assert 0


def benchmark(FLAGS, dataset_name, trainset, testset, paraphrase_set):

    paraphrase_strategy = get_strategy(FLAGS, FLAGS.strategy)
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
                              gpt2_gpu=FLAGS.gpt2_gpu,
                              bert_gpu=FLAGS.bert_gpu,
                              use_gpu=FLAGS.use_gpu)

    customize_metric = {
        "FlipPred_sim0.95_ppl2": flip_pred_agg_fn(use_sim=0.95, ppl_score=2),
        "FlipPred_sim0.90_ppl5": flip_pred_agg_fn(use_sim=0.90, ppl_score=5)
    }
    aggregate_measurements("RandomStrategy", G_EXP_NAME, results, customize_metric,
                           get_dataset_result_filename(dataset_name))


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
