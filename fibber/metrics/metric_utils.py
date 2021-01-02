import datetime
import json

import numpy as np
import pandas as pd
import tqdm

from fibber import log
from fibber.metrics.bert_classifier import BertClassifier
from fibber.metrics.classifier_base import ClassifierBase
from fibber.metrics.edit_distance_metric import EditDistanceMetric
from fibber.metrics.glove_semantic_similarity_metric import GloVeSemanticSimilarityMetric
from fibber.metrics.gpt2_grammar_quality_metric import GPT2GrammarQualityMetric
from fibber.metrics.metric_base import MetricBase
from fibber.metrics.use_semantic_similarity_metric import USESemanticSimilarityMetric

logger = log.setup_custom_logger(__name__)


DIRECTION_HIGHER_BETTER = "(↑)"
DIRECTION_LOWER_BETTER = "(↓)"
DIRECTION_UNKNOWN = "(x)"


class MetricBundle(object):
    """MetricBundle can help easily initialize and compute multiple metrics."""

    def __init__(self,
                 enable_edit_distance=True,
                 enable_use_semantic_similarity=True,
                 enable_glove_semantic_similarity=True,
                 enable_gpt2_grammar_quality=True,
                 enable_bert_clf_prediction=False,
                 **kargs):
        """Initialize various metrics.

        Args:
            enable_edit_distance (bool): whether to use editing distance in the bundle.
            enable_use_semantic_similarity (bool): whether to use Universal sentence encoder to
                compute sentence similarity
            enable_glove_semantic_similarity (bool): whether to use Glove embeddings to compute
                sentence similarity.
            enable_gpt2_grammar_quality (bool): whether to use GPT2 to compute sentence quality.
            enable_bert_clf_prediction (bool): whether to include BERT classifier prediction in
                metrics.
            kargs: arguments for metrics. kargs will be passed to all metrics.
        """
        super(MetricBundle, self).__init__()

        self._metrics = {}
        self._classifiers = {}
        self._target_clf = None

        self._advanced_aggregation_fn = {}

        if enable_edit_distance:
            self.add_metric(EditDistanceMetric(**kargs), DIRECTION_HIGHER_BETTER)
        if enable_use_semantic_similarity:
            self.add_metric(USESemanticSimilarityMetric(**kargs), DIRECTION_HIGHER_BETTER)
        if enable_glove_semantic_similarity:
            self.add_metric(GloVeSemanticSimilarityMetric(**kargs), DIRECTION_HIGHER_BETTER)
        if enable_gpt2_grammar_quality:
            self.add_metric(GPT2GrammarQualityMetric(**kargs), DIRECTION_LOWER_BETTER)
        if enable_bert_clf_prediction:
            self.add_classifier(BertClassifier(**kargs), set_target_clf=True)

    def add_metric(self, metric, direction):
        """Add a customized metric to metric bundle.

        Args:
            metric (MetricBase): the metric object to add.
            direction (str): choose from ``DIRECTION_HIGHER_BETTER``, ``DIRECTION_HIGHER_BETTER``
                and ``DIRECTION_UNKNOWN``.
        """
        if not isinstance(metric, MetricBase):
            logger.error("%s is not an instance of MetricBase.", str(metric))
            raise RuntimeError
        if str(metric) in self._metrics:
            logger.error("Duplicate metric %s.", str(metric))
            raise RuntimeError
        self._metrics[str(metric)] = (metric, direction)

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
        return self._metrics[metric_name][0]

    def get_metric_direction(self, metric_name):
        """Returns the direction of a metric.

        Metric name is the class name of a metric.

        Raises assertion error if metric is not found.

        Args:
            metric_name: the name of the matric.
        Returns:
            (object): a metric object.
        """
        assert metric_name in self._metrics
        return self._metrics[metric_name][1]

    def get_metric_names(self):
        """"Returns all metric names in this metric bundle.

        Returns:
            list of str
        """
        return list(self._metrics.keys())

    def add_classifier(self, classifier_metric, set_target_clf=False):
        """Set a target classifier to attack.

        Args:
            classifier_metric (ClassifierBase): A classifier metric to be added.
            set_target_clf (bool): whether to set this classifier metric as target classifier.
        """
        assert isinstance(classifier_metric, ClassifierBase)
        self._classifiers[str(classifier_metric)] = classifier_metric
        self._target_clf = str(classifier_metric)
        if set_target_clf:
            self.set_target_classifier_by_name(str(classifier_metric))

    def get_classifier(self, classifier_name):
        """Returns the classifier in current metric bundle.

        Args:
            classifier_name (str): the name of the requested classifier.
        """
        return self._classifiers[classifier_name]

    def get_classifier_names(self):
        return list(self._classifiers.keys())

    def set_target_classifier_by_name(self, classifier_name):
        """Set a target classifier to attack.

        Args:
            classifier_name (str): set a classifier as target classifier.
        """
        assert isinstance(classifier_name, str)
        assert classifier_name in self._classifiers
        self._target_clf = classifier_name

    def get_target_classifier_name(self):
        """Return the name of the target classifier."""
        return self._target_clf

    def get_target_classifier(self):
        """Returns the classifier for attack."""
        return self.get_classifier(self.get_target_classifier_name())

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the results of all metrics in the bundle for one pair of text.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record (str): the data record.
            paraphrase_field (str): choose from "text0", "text1".

        Returns:
            (dict): a dict with metric name as key.
        """
        ret = {}
        for name in self.get_metric_names():
            metric = self.get_metric(name)
            ret[name] = metric.measure_example(origin, paraphrase, data_record, paraphrase_field)
        for name in self.get_classifier_names():
            classifier = self.get_classifier(name)
            ret[name] = classifier.measure_example(
                origin, paraphrase, data_record, paraphrase_field)
        return ret

    def measure_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0"):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (list): a list containing dict of metrics for each paraphrase.
        """
        ret = [{} for i in range(len(paraphrase_list))]
        for name in self.get_metric_names():
            metric = self.get_metric(name)
            result = metric.measure_batch(origin, paraphrase_list, data_record, paraphrase_field)
            for i in range(len(paraphrase_list)):
                ret[i][name] = result[i]
        for name in self.get_classifier_names():
            classifier = self.get_classifier(name)
            result = classifier.measure_batch(origin, paraphrase_list,
                                              data_record, paraphrase_field)
            for i in range(len(paraphrase_list)):
                ret[i][name] = result[i]
        return ret

    def measure_dataset(self, results, output_filename):
        """Compute the all metrics for results on a dataset.

        Args:
            results (dict): A fibber dataset with paraphrase_list.
            output_filename (str): A json filename to store results and metrics.

        Returns:
            (dict): the results dict with ``original_text_metrics`` and ``paraphrase_metrics``
                added.
        """
        last_output_save_time = -1
        logger.info("Start measuring.")
        paraphrase_field = results["paraphrase_field"]

        for data_record in tqdm.tqdm(results["data"]):
            data_record_tmp = dict([(k, v) for k, v in data_record.items()
                                    if "_paraphrases" not in k])

            # Run metrics on original text
            data_record["original_text_metrics"] = self.measure_example(
                data_record[paraphrase_field], data_record[paraphrase_field],
                data_record_tmp, paraphrase_field)

            # Run metrics on paraphrased text
            data_record["paraphrase_metrics"] = self.measure_batch(
                data_record[paraphrase_field], data_record[paraphrase_field + "_paraphrases"],
                data_record_tmp, paraphrase_field)

            # save tmp output every 30 seconds
            if datetime.datetime.now().timestamp() - last_output_save_time > 30:
                with open(output_filename, "w") as f:
                    json.dump(results, f, indent=2)
                datetime.datetime.now().timestamp()

        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def add_advanced_aggregation_fn(self, aggregation_name, aggregation_fn, direction):
        """Add advanced aggregation function.

        Some aggregation function can aggregate multiple metrics, these aggregation functions
        can be added here.

        Args:
            aggregation_name (str): the name of the aggregation.
            aggregation_fn (fn): an aggregation function that takes ``data_record`` as arg.
            direction (str): chose from DIRECTION_HIGHER_BETTER, DIRECTION_LOWER_BETTER,
                and DIRECTION_UNKNOWN.
        """
        if aggregation_name in self._advanced_aggregation_fn:
            logger.error("Duplicate advanced aggregation function %s.", aggregation_name)
            raise RuntimeError
        self._advanced_aggregation_fn[aggregation_name] = (aggregation_fn, direction)

    def get_advanced_aggregation_fn_names(self):
        return list(self._advanced_aggregation_fn.keys())

    def get_advanced_aggregation_fn_direction(self, aggregation_name):
        return self._advanced_aggregation_fn[aggregation_name][1]

    def get_advanced_aggregation_fn(self, aggregation_name):
        return self._advanced_aggregation_fn[aggregation_name][0]

    def aggregate_metrics(self, dataset_name, paraphrase_strategy_name, experiment_name, results):
        """Aggregate paraphrase metrics on a dataset to a summary and store in a dict.

        Args:
            dataset_name (str): the name of the dataset.
            paraphrase_strategy_name (str): the name of the paraphrase strategy.
            experiment_name (str): the name of the experiment.
            results (dict): the fibber dataset with paraphrases and metrics. The return value of
                ``compute_metrics``.

        Returns:
            (dict): the aggregated metrics.
        """
        aggregated_result = pd.DataFrame()
        for data_record in results["data"]:
            aggregate_result_tmp = {}
            metric_df = pd.DataFrame(data_record["paraphrase_metrics"])
            for metric_name in metric_df.columns.tolist():
                if metric_name in self.get_classifier_names():
                    continue
                metric_value = metric_df[metric_name].values
                direction = self.get_metric_direction(metric_name)

                # aggregate mean
                agg_value = np.mean(metric_value)
                aggregate_result_tmp[metric_name + direction] = float(agg_value)

                # aggregate std
                agg_value = np.std(metric_value)
                aggregate_result_tmp[metric_name + "_std"] = float(agg_value)

            aggregate_result_tmp["ParaphrasesPerExample"] = len(data_record["paraphrase_metrics"])

            for agg_name in self.get_advanced_aggregation_fn_names():
                agg_fn = self.get_advanced_aggregation_fn(agg_name)
                direction = self.get_advanced_aggregation_fn_direction(agg_name)
                aggregate_result_tmp[agg_name + direction] = agg_fn(data_record)

            aggregated_result = aggregated_result.append(aggregate_result_tmp, ignore_index=True)

        aggregated_result = dict(aggregated_result.mean(skipna=True))
        # hack column order by adding 0
        aggregated_result["0_dataset_name"] = dataset_name
        aggregated_result["1_paraphrase_strategy_name"] = paraphrase_strategy_name
        aggregated_result["2_experiment_name"] = experiment_name

        return aggregated_result
