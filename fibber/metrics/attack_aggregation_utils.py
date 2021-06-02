"""This module defines customized metric aggregation functions."""

import itertools
import math
from multiprocessing import Pool

import numpy as np

from fibber.metrics.edit_distance_metric import EditDistanceMetric
from fibber.metrics.metric_utils import DIRECTION_HIGHER_BETTER, DIRECTION_LOWER_BETTER


def paraphrase_classification_accuracy_agg_fn_constructor(target_clf):
    """This function makes a aggregation function for the target classification metric.

    The aggregation function outputs the after attack accuracy of the BERT classifier.

    Args:
        target_clf (str): the metric name of the target classifier.
    """

    def agg_fn(data_record):
        if data_record["original_text_metrics"][target_clf] != data_record["label"]:
            return 0
        for item in data_record["paraphrase_metrics"]:
            if (item[target_clf] != data_record["label"]):
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


def get_best_adv_by_metric(data_record, target_clf, metric_name, lower_better):
    """Find the best adversarial example by some metric.

    Args:
        data_record (dict): a data record with paraphrases and metrics.
        target_clf (str): the targeted classifier.
        metric_name (str): the metric to pick the best adversarial example.
        lower_better (bool): if true, find the adversarial example with the smallest metric value.
    Returns:
         (dict): the metrics of the best adversarial example. None if no legitimate adversarial
            example is found or the original sentence is misclassified.
    """
    if data_record["original_text_metrics"][target_clf] != data_record["label"]:
        return None
    best_score = None
    best_metrics = None
    for metrics in data_record["paraphrase_metrics"]:
        if metrics[target_clf] == data_record["label"]:
            continue
        if best_score is None or ((lower_better and metrics[metric_name] < best_score)
                                  or (not lower_better and metrics[metric_name] > best_score)):
            best_score = metrics[metric_name]
            best_metrics = metrics
    return best_metrics


def get_best_adv_metric_fn_constructor(get_best_adv_fn, metric_name, target_clf):
    """Returns an aggregation function that extracts the value of a specified metric for the best
    adversarial example.

    The aggregation function returns NaN if no legitimate adversarial example is found.

    Args:
        get_best_adv_fn (fn): a function that returns the metric dict of the best adversarial
            example.
        metric_name (str): a metric name.
        target_clf (str): the targeted classifier.
    Returns:
        (fn): an aggregation function that takes data_record as an input.
    """
    def agg_fn(data_record):
        best_metrics = get_best_adv_fn(data_record, target_clf)
        if best_metrics is not None:
            return best_metrics[metric_name]
        return math.nan
    return agg_fn


def add_sentence_level_adversarial_attack_metrics(metric_bundle,
                                                  best_adv_metric_name,
                                                  best_adv_metric_lower_better):
    """Add advanced aggregation functions related to adversarial attack to a specific metric
    bundle.

    Args:
        metric_bundle (MetricBundle): a metric bundle object to add aggregation functions.
        best_adv_metric_name (str):
        best_adv_metric_lower_better (bool):
    """
    metric_bundle.add_advanced_aggregation_fn(
        "PairwiseEditDistance",
        pairwise_editing_distance_fn,
        DIRECTION_HIGHER_BETTER
    )

    target_clf = metric_bundle.get_target_classifier_name()

    metric_bundle.add_advanced_aggregation_fn(
            "%s_AfterAttackAccuracy" % target_clf,
            paraphrase_classification_accuracy_agg_fn_constructor(target_clf),
            DIRECTION_LOWER_BETTER
    )

    for metric_name in metric_bundle.get_metric_names():
        metric_bundle.add_advanced_aggregation_fn(
            "best_adv_%s" % metric_name,
            get_best_adv_metric_fn_constructor(
                lambda _data_record, _target_clf: get_best_adv_by_metric(
                    _data_record, _target_clf, best_adv_metric_name, best_adv_metric_lower_better),
                metric_name, metric_bundle.get_target_classifier_name()),
            metric_bundle.get_metric_direction(metric_name)
        )
