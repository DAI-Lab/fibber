"""This module defines customized metric aggregation functions."""

import itertools
import math
from multiprocessing import Pool

import numpy as np

from fibber.metrics.edit_distance_metric import EditDistanceMetric
from fibber.metrics.metric_utils import DIRECTION_HIGHER_BETTER, DIRECTION_LOWER_BETTER


def paraphrase_classification_accuracy_agg_fn_constructor(
        gpt2_ppl_threshold, use_sim_threshold, target_clf):
    """This function makes a aggregation function for the target classification metric.

    The aggregation function outputs the after attack accuracy of the BERT classifier.

    Args:
        gpt2_ppl_threshold (float): the threshold for ``GPT2GrammarQualityMetric`` metric. The
            adversarial example should have the perplexity ratio measured by GPT2 lower than
            this threshold.
        use_sim_threshold (float): the threshold for ``USESemanticSimilarityMetric`` metric. The
            adversarial example should have the USE similarity higher than this threshold.
        target_clf (str): the metric name of the target classifier.
    """

    def agg_fn(data_record):
        if data_record["original_text_metrics"][target_clf] != data_record["label"]:
            return 0
        for item in data_record["paraphrase_metrics"]:
            if (item[target_clf] != data_record["label"]
                    and item["GPT2GrammarQualityMetric"] < gpt2_ppl_threshold
                    and item["USESemanticSimilarityMetric"] > use_sim_threshold):
                return 0
        return 1

    return agg_fn


def editing_distance_element_worker(x):
    editing_distance_metric = EditDistanceMetric()
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

    distance = pool.map(editing_distance_element_worker, tuples)
    return float(np.mean(distance))


def get_best_adv_by_sim(data_record, target_clf, gpt2_ppl_threshold=5, use_sim_threshold=0.85):
    """Find the adversarial example with best similarity.

    Args:
        data_record (dict): a data record with paraphrases and metrics.
        target_clf (str): the targeted classifier.
        gpt2_ppl_threshold (float): the gpt2 perplexity ration threshold for a legitimate
            adversarial example.
        use_sim_threshold (float): the USE cosine similarity threshold for a legitimate adversarial
            example.
    Returns:
         (dict): the metrics of the best adversarial example. None if no legitimate adversarial
            example is found.
    """
    if data_record["original_text_metrics"][target_clf] != data_record["label"]:
        return None
    best_score = -1
    best_metrics = None
    for metrics in data_record["paraphrase_metrics"]:
        if (metrics[target_clf] == data_record["label"]
                or metrics["GPT2GrammarQualityMetric"] > gpt2_ppl_threshold
                or metrics["USESemanticSimilarityMetric"] < use_sim_threshold):
            continue
        if metrics["USESemanticSimilarityMetric"] > best_score:
            best_score = metrics["USESemanticSimilarityMetric"]
            best_metrics = metrics
    return best_metrics


def get_best_adv_by_ppl(data_record, target_clf, gpt2_ppl_threshold=5, use_sim_threshold=0.85):
    """Find the adversarial example with lowest perplexity.

    Args:
        data_record (dict): a data record with paraphrases and metrics.
        target_clf (str): the targeted classifier.
        gpt2_ppl_threshold (float): the gpt2 perplexity ration threshold for a legitimate
            adversarial example.
        use_sim_threshold (float): the USE cosine similarity threshold for a legitimate adversarial
            example.
    Returns:
         (dict): the metrics of the best adversarial example. None if no legitimate adversarial
            example is found.
    """
    if data_record["original_text_metrics"][target_clf] != data_record["label"]:
        return None
    best_score = 1e8
    best_metrics = None
    for metrics in data_record["paraphrase_metrics"]:
        if (metrics[target_clf] == data_record["label"]
                or metrics["GPT2GrammarQualityMetric"] > gpt2_ppl_threshold
                or metrics["USESemanticSimilarityMetric"] < use_sim_threshold):
            continue
        if metrics["GPT2GrammarQualityMetric"] < best_score:
            best_score = metrics["GPT2GrammarQualityMetric"]
            best_metrics = metrics
    return best_metrics


def get_best_adv_metric_fn_constructor(get_best_adv_fn, metric_name, target_clf,
                                       gpt2_ppl_threshold=5, use_sim_threshold=0.85):
    """Returns an aggregation function that extracts the value of a specified metric for the best
    adversarial example.

    The aggregation function returns NaN if no legitimate adversarial example is found.

    Args:
        get_best_adv_fn (fn): a function that returns the metric dict of the best adversarial
            example.
        metric_name (str): a metric name.
        target_clf (str): the targeted classifier.
        gpt2_ppl_threshold (float): the gpt2 perplexity ration threshold for a legitimate
            adversarial example.
        use_sim_threshold (float): the USE cosine similarity threshold for a legitimate adversarial
            example.
    Returns:
        (fn): an aggregation function that takes data_record as an input.
    """
    def agg_fn(data_record):
        best_metrics = get_best_adv_fn(data_record, target_clf,
                                       gpt2_ppl_threshold, use_sim_threshold)
        if best_metrics is not None:
            return best_metrics[metric_name]
        return math.nan

    return agg_fn


def add_sentence_level_adversarial_attack_metrics(
        metric_bundle, gpt2_ppl_threshold, use_sim_threshold):
    """Add advanced aggregation functions related to adversarial attack to a specific metric
    bundle.

    Args:
        metric_bundle (MetricBundle): a metric bundle object to add aggregation functions.
        gpt2_ppl_threshold (float): the gpt2 perplexity ration threshold for a legitimate
            adversarial example.
        use_sim_threshold (float): the USE cosine similarity threshold for a legitimate adversarial
            example.
    """
    metric_bundle.add_advanced_aggregation_fn(
        "PairwiseEditDistance",
        pairwise_editing_distance_fn,
        DIRECTION_HIGHER_BETTER
    )

    for classifier_name in metric_bundle.get_classifier_names():
        metric_bundle.add_advanced_aggregation_fn(
            "3_%s_AfterAttackAccuracy_USE_%.2f_GPT_%.1f" % (
                classifier_name, use_sim_threshold, gpt2_ppl_threshold),
            paraphrase_classification_accuracy_agg_fn_constructor(
                gpt2_ppl_threshold=gpt2_ppl_threshold,
                use_sim_threshold=use_sim_threshold,
                target_clf=classifier_name),
            DIRECTION_LOWER_BETTER
        )

        for metric_name in metric_bundle.get_metric_names():
            metric_bundle.add_advanced_aggregation_fn(
                "%s_best_sim_adv_%s" % (classifier_name, metric_name),
                get_best_adv_metric_fn_constructor(
                    get_best_adv_by_sim, metric_name, metric_bundle.get_target_classifier_name()),
                metric_bundle.get_metric_direction(metric_name)
            )
