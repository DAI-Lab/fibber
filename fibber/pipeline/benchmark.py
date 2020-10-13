import argparse
import datetime
import os

from .. import log
from ..dataset.dataset_utils import get_dataset, subsample_dataset
from ..measurement.measurement_utils import (
    MeasurementBundle, aggregate_measurements, measure_quality)
from ..resource_utils import update_detailed_result
from ..strategy.identical_strategy import IdenticalStrategy
from ..strategy.random_strategy import RandomStrategy
from ..strategy.textfooler_strategy import TextFoolerStrategy
from ..strategy.gibbs_sampling_strategy import GibbsSamplingStrategy
from ..strategy.gibbs_sampling_x_strategy import GibbsSamplingXStrategy
from ..strategy.gibbs_sampling_wpe_strategy import GibbsSamplingWPEStrategy

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ag")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
parser.add_argument("--subsample_testset", type=int, default=1000)

parser.add_argument("--strategy", type=str, default="RandomStrategy")
parser.add_argument("--strategy_gpu", type=int, default=-1)
# measurement args
parser.add_argument("--gpt2_gpu", type=int, default=-1)
parser.add_argument("--bert_gpu", type=int, default=-1)
parser.add_argument("--use_gpu", type=int, default=-1)
parser.add_argument("--bert_clf_steps", type=int, default=20000)

RandomStrategy.add_parser_args(parser)
IdenticalStrategy.add_parser_args(parser)

G_EXP_NAME = None


def get_output_filename(FLAGS, prefix="", suffix=""):
    global G_EXP_NAME
    if G_EXP_NAME is None:
        G_EXP_NAME = (FLAGS.dataset + "-" + FLAGS.strategy + "-"
                      + datetime.datetime.now().strftime("%m%d-%H%M%S"))
    return prefix + G_EXP_NAME + suffix


def paraphrase_pred_accuracy_agg_fn(use_sim, ppl_score):
    def agg_fn(data_record):
        if data_record["original_text_measurements"]["BertClfPrediction"] != data_record["label"]:
            return 0
        for item in data_record["paraphrase_measurements"]:
            if (item["BertClfPrediction"] != data_record["label"]
                and item["GPT2GrammarQuality"] < ppl_score
                    and item["USESemanticSimilarity"] > use_sim):
                return 0
        return 1
    return agg_fn


def get_strategy(FLAGS, strategy_name, measurement_bundle):
    if strategy_name == "RandomStrategy":
        return RandomStrategy(FLAGS, measurement_bundle)
    if strategy_name == "IdenticalStrategy":
        return IdenticalStrategy(FLAGS, measurement_bundle)
    if strategy_name == "TextFoolerStrategy":
        return TextFoolerStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingStrategy":
        return GibbsSamplingStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingXStrategy":
        return GibbsSamplingXStrategy(FLAGS, measurement_bundle)
    if strategy_name == "GibbsSamplingWPEStrategy":
        return GibbsSamplingWPEStrategy(FLAGS, measurement_bundle)
    else:
        assert 0


def benchmark(FLAGS, dataset_name, trainset, testset, paraphrase_set):
    logger.info("Build measurement bundle.")
    measurement_bundle = MeasurementBundle(
        use_glove_semantic_similarity=False,
        use_bert_clf_prediction=True,
        use_gpu_id=FLAGS.use_gpu, gpt2_gpu_id=FLAGS.gpt2_gpu,
        bert_gpu_id=FLAGS.bert_gpu, dataset_name=dataset_name,
        trainset=trainset, testset=testset,
        bert_clf_steps=FLAGS.bert_clf_steps)

    paraphrase_strategy = get_strategy(FLAGS, FLAGS.strategy, measurement_bundle)
    paraphrase_strategy.fit(trainset)

    tmp_output_filename = os.path.join(
        FLAGS.output_dir, get_output_filename(FLAGS, suffix="-tmp.json"))
    logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
    results = paraphrase_strategy.paraphrase(
        paraphrase_set, FLAGS.num_paraphrases_per_text, tmp_output_filename)

    output_filename = os.path.join(
        FLAGS.output_dir, get_output_filename(FLAGS, suffix="-with-measurement.json"))

    results = measure_quality(measurement_bundle, results=results,
                              paraphrase_field=paraphrase_set["paraphrase_field"],
                              output_filename=output_filename)

    customize_metric = {
        "3_ParaphraseAcc_usesim0.90_ppl2":
            paraphrase_pred_accuracy_agg_fn(use_sim=0.90, ppl_score=2),
        "4_ParaphraseAcc_usesim0.85_ppl5":
            paraphrase_pred_accuracy_agg_fn(use_sim=0.85, ppl_score=5)
    }
    aggregated_result = aggregate_measurements(
        dataset_name, str(paraphrase_strategy), G_EXP_NAME, results, customize_metric)
    update_detailed_result(aggregated_result)


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    assert FLAGS.output_dir is not None
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "log"), exist_ok=True)

    log.add_filehandler(
        logger, os.path.join(FLAGS.output_dir, "log", get_output_filename(FLAGS, suffix=".log")))
    log.remove_logger_tf_handler(logger)

    trainset, testset = get_dataset(FLAGS.dataset)
    paraphrase_set = subsample_dataset(testset, FLAGS.subsample_testset)
    logger.info("Subsample test set to %d.", FLAGS.subsample_testset)
    benchmark(FLAGS, FLAGS.dataset, trainset, testset, paraphrase_set)
