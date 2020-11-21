"""This module defines customized metric aggregation functions."""

import itertools
import math
from multiprocessing import Pool

import numpy as np

from fibber.metrics.editing_distance import EditingDistance

DEFAULT_USE_SIM_THRESHOLD = 0.85
DEFAULT_GPT2_PPL_THRESHOLD = 5


def paraphrase_classification_accuracy_agg_fn(use_sim, ppl_score):
    """This function makes a aggregation function for BERT classification metric.

    The aggregation function outputs the after attack accuracy of the BERT classifier.

    Args:
        use_sim (float): the threshold for ``USESemanticSimilarity`` metric. The adversarial
            example should have the USE similarity higher than this threshold.
        ppl_score (float): the threshold for ``GPT2GrammarQuality`` metric. The adversarial example
            should have the perplexity ratio measured by GPT2 lower than this threshold.
    """

    def agg_fn(data_record):
        if data_record["original_text_metrics"]["BertClfPrediction"] != data_record["label"]:
            return 0
        for item in data_record["paraphrase_metrics"]:
            if (item["BertClfPrediction"] != data_record["label"]
                    and item["GPT2GrammarQuality"] < ppl_score
                    and item["USESemanticSimilarity"] > use_sim):
                return 0
        return 1

    return agg_fn


def editing_distance_element_fn(x):
    editing_distance_metric = EditingDistance()
    return editing_distance_metric.measure_example(x[0], x[1])


pool = Pool(8)


def pairwise_editing_distance_fn(data_record):
    """Compute the average pairwise editing distance metric."""
    global pool
    paraphrases = None
    for k, v in data_record.items():
        if k in ["text0_paraphrases", "text1_paraphrases"]:
            paraphrases = v
            break

    assert paraphrases is not None

    tuples = list(itertools.combinations(paraphrases, 2))

    distance = pool.map(editing_distance_element_fn, tuples)
    return float(np.mean(distance))


def get_best_adv_by_sim(data_record):
    """Find the adversarial example with best similarity.

    Args:
        data_record (dict): a data record with paraphrases and metrics.
    Returns:
         (dict): the metrics of the best adversarial example. None if no legitimate adversarial
            example is found.
    """
    if data_record["original_text_metrics"]["BertClfPrediction"] != data_record["label"]:
        return None
    best_score = -1
    best_metrics = None
    for metrics in data_record["paraphrase_metrics"]:
        if (metrics["BertClfPrediction"] == data_record["label"]
                or metrics["GPT2GrammarQuality"] > DEFAULT_GPT2_PPL_THRESHOLD
                or metrics["USESemanticSimilarity"] < DEFAULT_USE_SIM_THRESHOLD):
            continue
        if metrics["USESemanticSimilarity"] > best_score:
            best_score = metrics["USESemanticSimilarity"]
            best_metrics = metrics
    return best_metrics


def get_best_adv_by_ppl(data_record):
    """Find the adversarial example with lowest perplexity.

    Args:
        data_record (dict): a data record with paraphrases and metrics.
    Returns:
         (dict): the metrics of the best adversarial example. None if no legitimate adversarial
            example is found.
    """
    if data_record["original_text_metrics"]["BertClfPrediction"] != data_record["label"]:
        return None
    best_score = 1e8
    best_metrics = None
    for metrics in data_record["paraphrase_metrics"]:
        if (metrics["BertClfPrediction"] == data_record["label"]
                or metrics["GPT2GrammarQuality"] > DEFAULT_GPT2_PPL_THRESHOLD
                or metrics["USESemanticSimilarity"] < DEFAULT_USE_SIM_THRESHOLD):
            continue
        if metrics["GPT2GrammarQuality"] < best_score:
            best_score = metrics["GPT2GrammarQuality"]
            best_metrics = metrics
    return best_metrics


def get_best_adv_metric_fn(get_best_adv_fn, metric_name):
    """Returns an aggregation function that extracts the value of a specified metric for the best
    adversarial example.

    The aggregation function returns NaN if no legitimate adversarial example is found.

    Args:
        get_best_adv_fn (fn): a function that returns the metric dict of the best adversarial
            example.
        metric_name (str): a metric name.
    Returns:
        (fn): an aggregation function that takes data_record as an input.
    """
    def agg_fn(data_record):
        best_metrics = get_best_adv_fn(data_record)
        if best_metrics is not None:
            return best_metrics[metric_name]
        return math.nan

    return agg_fn


customized_metric_aggregation_fn_dict = {
    "3_ParaphraseAcc_use0.90_ppl2":
        paraphrase_classification_accuracy_agg_fn(use_sim=0.90, ppl_score=2),
    "4_ParaphraseAcc_use0.85_ppl5":
        paraphrase_classification_accuracy_agg_fn(use_sim=0.85, ppl_score=5),
    "pairwise_edit_distance": pairwise_editing_distance_fn,
    "best_sim_adv_GPT2GrammarQuality":
        get_best_adv_metric_fn(get_best_adv_by_sim, "GPT2GrammarQuality"),
    "best_sim_adv_USESemanticSimilarity":
        get_best_adv_metric_fn(get_best_adv_by_sim, "USESemanticSimilarity"),
    "best_sim_adv_EditingDistance":
        get_best_adv_metric_fn(get_best_adv_by_sim, "EditingDistance"),
    "best_sim_adv_GloveSemanticSimilarity":
        get_best_adv_metric_fn(get_best_adv_by_sim, "GloVeSemanticSimilarity"),
    "best_ppl_adv_GPT2GrammarQuality":
        get_best_adv_metric_fn(get_best_adv_by_ppl, "GPT2GrammarQuality"),
    "best_ppl_adv_USESemanticSimilarity":
        get_best_adv_metric_fn(get_best_adv_by_ppl, "USESemanticSimilarity"),
    "best_ppl_adv_EditingDistance":
        get_best_adv_metric_fn(get_best_adv_by_ppl, "EditingDistance"),
    "best_ppl_adv_GloveSemanticSimilarity":
        get_best_adv_metric_fn(get_best_adv_by_ppl, "GloVeSemanticSimilarity"),
}

customized_metric_for_nun_wins = [
    ("3_ParaphraseAcc_use0.90_ppl2", "L"),
    ("4_ParaphraseAcc_use0.85_ppl5", "L"),
    ("best_sim_adv_USESemanticSimilarity", "H"),
    ("best_sim_adv_GPT2GrammarQuality", "L"),
    ("best_ppl_adv_USESemanticSimilarity", "H"),
    ("best_ppl_adv_GPT2GrammarQuality", "L"),
]
