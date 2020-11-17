import datetime
import json

import numpy as np
import pandas as pd
import tqdm

from fibber import log
from fibber.metrics.bert_clf_prediction import BertClfPrediction
from fibber.metrics.editing_distance import EditingDistance
from fibber.metrics.glove_semantic_similarity import GloVeSemanticSimilarity
from fibber.metrics.gpt2_grammar_quality import GPT2GrammarQuality
from fibber.metrics.metric_base import MetricBase
from fibber.metrics.use_semantic_similarity import USESemanticSimilarity

logger = log.setup_custom_logger(__name__)


MEAN_AGG = "mean"
STD_AGG = "std"

SPECIAL_METRIC_AGGREGATION = {
    "BertClfPrediction": []
}
DEFAULT_AGGREGATION = [MEAN_AGG, STD_AGG]


def mean_aggregation_fn(x):
    """Compute mean of a list."""
    return float(np.mean(x))


def std_aggregation_fn(x):
    """Compute std of a list."""
    return float(np.std(x))


AGGREGATION_NAME_TO_FN = {
    MEAN_AGG: mean_aggregation_fn,
    STD_AGG: std_aggregation_fn
}


class MetricBundle(object):
    """MetricBundle can help easily initialize and compute multiple metrics."""

    def __init__(self,
                 use_editing_distance=True,
                 use_use_semantic_similarity=True,
                 use_glove_semantic_similarity=True,
                 use_gpt2_grammar_quality=True,
                 use_bert_clf_prediction=False,
                 customized_metrics=[],
                 **kargs):
        """Initialize various metrics.

        Args:
            use_editing_distance (bool): whether to use editing distance in the bundle.
            use_use_semantic_similarity (bool): whether to use Universal sentence encoder to
                compute sentence similarity
            use_glove_semantic_similarity (bool): whether to use Glove embeddings to compute
                sentence similarity.
            use_gpt2_grammar_quality (bool): whether to use GPT2 to compute sentence quality.
            use_bert_clf_prediction (bool): whether to include BERT classifier prediction in
                metrics.
            customized_metrics (list): a list of customized metrics.
            kargs: arguments for metrics. kargs will be passed to all metrics.
        """
        super(MetricBundle, self).__init__()

        assert isinstance(customized_metrics, list)
        for item in customized_metrics:
            assert isinstance(item, MetricBase)

        self._metrics = {}

        if use_editing_distance:
            metric = EditingDistance(**kargs)
            self._metrics[str(metric)] = metric
        if use_use_semantic_similarity:
            metric = USESemanticSimilarity(**kargs)
            self._metrics[str(metric)] = metric
        if use_glove_semantic_similarity:
            metric = GloVeSemanticSimilarity(**kargs)
            self._metrics[str(metric)] = metric
        if use_gpt2_grammar_quality:
            metric = GPT2GrammarQuality(**kargs)
            self._metrics[str(metric)] = metric
        if use_bert_clf_prediction:
            metric = BertClfPrediction(**kargs)
            self._metrics[str(metric)] = metric

        for metric in customized_metrics:
            assert str(metric) not in self._metrics, "Duplicate metric name."
            self._metrics[str(metric)] = metric

    def get_metric(self, metric_name):
        """Returns a metric in the bundle using the metric name.

        Metric name is the class name of a metric.

        Raises assertion error if metric is not found.

        Args:
            metric_name: the name of the matric.
        Returns:
            (object): a metric object.
        """
        assert metric_name in self._metrics
        return self._metrics[metric_name]

    def get_classifier_for_attack(self):
        """Returns the classifier for attack."""
        return self.get_metric("BertClfPrediction")

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the results of all metrics in the bundle for one pair of text.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record (str): the data record.
            paraphrase_field (str): choose from "text0", "text1".

        Returns:
            (dict):
        """
        ret = {}
        for name, metric in self._metrics.items():
            ret[name] = metric.measure_example(origin, paraphrase, data_record, paraphrase_field)
        return ret


def compute_metrics(metric_bundle, results, output_filename):
    """Compute the all metrics for results on a dataset.

    Args:
        metric_bundle (object): a MetricBundle object.
        results (dict): A fibber dataset with paraphrases.
        output_filename (str): A json filename to store results and metrics.

    Returns:
        (dict): the results dict with ``original_text_metrics`` and ``paraphrase_metrics`` added.
    """
    last_output_save_time = -1
    logger.info("Start measuring.")
    paraphrase_field = results["paraphrase_field"]

    for data_record in tqdm.tqdm(results["data"]):
        data_record_tmp = dict([(k, v) for k, v in data_record.items() if "_paraphrases" not in k])

        # Run metrics on original text
        data_record["original_text_metrics"] = metric_bundle.measure_example(
            data_record[paraphrase_field], data_record[paraphrase_field],
            data_record_tmp, paraphrase_field)

        # Run metrics on paraphrased text
        paraphrase_metric_list = []
        for paraphrase in data_record[paraphrase_field + "_paraphrases"]:
            paraphrase_metric_list.append(
                metric_bundle.measure_example(
                    data_record[paraphrase_field], paraphrase, data_record_tmp, paraphrase_field))

        data_record["paraphrase_metrics"] = paraphrase_metric_list

        # save tmp output every 30 seconds
        if datetime.datetime.now().timestamp() - last_output_save_time > 30:
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
            datetime.datetime.now().timestamp()

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_metrics(dataset_name, paraphrase_strategy_name, experiment_name, results,
                      customize_aggregation_fn_dict):
    """Aggregate paraphrase metrics on a dataset to a summary and store in a dict.

    Args:
        dataset_name (str): the name of the dataset.
        paraphrase_strategy_name (str): the name of the paraphrase strategy.
        experiment_name (str): the name of the experiment.
        results (dict): the fibber dataset with paraphrases and metrics. The return value of
            ``compute_metrics``.
        customize_aggregation_fn_dict (dict): A dict of customized aggregations. The dict is
            a mapping from aggregation name to aggregation function. The aggregation function
            should take one data_record, and returns a float.

    Returns:
        (dict): the aggregated metrics.
    """
    aggregated_result = pd.DataFrame()
    for data_record in results["data"]:
        aggregate_result_tmp = {}
        metric_df = pd.DataFrame(data_record["paraphrase_metrics"])
        for metric_name in metric_df.columns.tolist():
            metric_value = metric_df[metric_name].values
            if metric_name in SPECIAL_METRIC_AGGREGATION:
                aggregation_list = SPECIAL_METRIC_AGGREGATION[metric_name]
            else:
                aggregation_list = DEFAULT_AGGREGATION

            for aggregation in aggregation_list:
                aggregation_fn = AGGREGATION_NAME_TO_FN[aggregation]
                aggregated_value = aggregation_fn(metric_value)
                aggregate_result_tmp[metric_name + "_" + aggregation] = aggregated_value

        aggregate_result_tmp["ParaphrasesPerExample"] = len(data_record["paraphrase_metrics"])

        for name, fn in customize_aggregation_fn_dict.items():
            aggregate_result_tmp[name] = fn(data_record)

        aggregated_result = aggregated_result.append(aggregate_result_tmp, ignore_index=True)

    aggregated_result = dict(aggregated_result.mean(skipna=True))
    # hack column order by adding 0
    aggregated_result["0_dataset_name"] = dataset_name
    aggregated_result["1_paraphrase_strategy_name"] = paraphrase_strategy_name
    aggregated_result["2_experiment_name"] = experiment_name

    return aggregated_result
