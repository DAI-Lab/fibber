import datetime
import json

import numpy as np
import pandas as pd
import tqdm

from fibber import log
from fibber.metrics.classifier.classifier_base import ClassifierBase
from fibber.metrics.classifier.fasttext_classifier import FasttextClassifier
from fibber.metrics.classifier.transformer_classifier import TransformerClassifier
from fibber.metrics.distance.edit_distance_metric import EditDistanceMetric
from fibber.metrics.distance.ref_bleu_metric import RefBleuMetric
from fibber.metrics.distance.self_bleu_metric import SelfBleuMetric
from fibber.metrics.fluency.bert_perplexity_metric import BertPerplexityMetric
from fibber.metrics.fluency.gpt2_perplexity_metric import GPT2PerplexityMetric
from fibber.metrics.metric_base import MetricBase
from fibber.metrics.similarity.ce_similarity_metric import CESimilarityMetric
from fibber.metrics.similarity.glove_similarity_metric import GloVeSimilarityMetric
from fibber.metrics.similarity.use_similarity_metric import USESimilarityMetric

logger = log.setup_custom_logger(__name__)


DIRECTION_HIGHER_BETTER = "(↑)"
DIRECTION_LOWER_BETTER = "(↓)"
DIRECTION_UNKNOWN = "(x)"


class MetricBundle(object):
    """MetricBundle can help easily initialize and compute multiple metrics."""

    def __init__(self,
                 enable_edit_distance=True,
                 enable_use_similarity=True,
                 enable_glove_similarity=True,
                 enable_gpt2_perplexity=False,
                 enable_transformer_classifier=True,
                 enable_ce_similarity=False,
                 enable_fasttext_classifier=False,
                 enable_bert_perplexity=True,
                 enable_bert_perplexity_per_class=False,
                 enable_self_bleu=False,
                 enable_ref_bleu=False,
                 target_clf="transformer",
                 field="text0", bs=32, **kwargs):
        """Initialize various metrics.

        Args:
            enable_edit_distance (bool): whether to use editing distance in the bundle.
            enable_use_similarity (bool): whether to use Universal sentence encoder to
                compute sentence similarity
            enable_glove_similarity (bool): whether to use Glove embeddings to compute
                sentence similarity.
            enable_gpt2_perplexity (bool): whether to use GPT2 to compute sentence quality.
            enable_transformer_classifier (bool): whether to include BERT classifier prediction in
                metrics.
            enable_ce_similarity (bool): whether to use Cross Encoder to measure sentence
                similarity.
            enable_fasttext_classifier (bool): whether to include Fasttext classifier prediction
                in metrics.
            target_clf (str): choose from "trasformer", "fasttext".
            field (str): the field where perturbation can happen.
            bs (int): batch size.
            kwargs: arguments for metrics. kwargs will be passed to all metrics.
        """
        super(MetricBundle, self).__init__()

        self._metrics = {}
        self._classifiers = {}
        self._target_clf = None
        self._field = field

        self._advanced_aggregation_fn = {}

        if enable_edit_distance:
            self.add_metric(EditDistanceMetric(field=field, bs=bs, **kwargs),
                            DIRECTION_UNKNOWN)
        if enable_use_similarity:
            self.add_metric(USESimilarityMetric(field=field, bs=bs, **kwargs),
                            DIRECTION_HIGHER_BETTER)
        if enable_glove_similarity:
            self.add_metric(GloVeSimilarityMetric(field=field, bs=bs, **kwargs),
                            DIRECTION_HIGHER_BETTER)
        if enable_gpt2_perplexity:
            self.add_metric(GPT2PerplexityMetric(field=field, bs=bs, **kwargs),
                            DIRECTION_LOWER_BETTER)
        if enable_ce_similarity:
            self.add_metric(CESimilarityMetric(field=field, bs=bs, **kwargs),
                            DIRECTION_HIGHER_BETTER)
        if enable_transformer_classifier:
            self.add_classifier(TransformerClassifier(field=field, bs=bs, **kwargs),
                                set_target_clf=(target_clf == "transformer"))
        if enable_fasttext_classifier:
            self.add_classifier(FasttextClassifier(field=field, bs=bs, **kwargs),
                                set_target_clf=(target_clf == "fasttext"))
        if enable_bert_perplexity:
            self.add_metric(BertPerplexityMetric(field=field, bs=bs, **kwargs),
                            DIRECTION_LOWER_BETTER)
        if enable_bert_perplexity_per_class:
            n_labels = len(kwargs["trainset"]["label_mapping"])
            for i in range(n_labels):
                self.add_metric(
                    BertPerplexityMetric(bert_ppl_filter=i, field=field, bs=bs, **kwargs),
                    DIRECTION_LOWER_BETTER)
        if enable_self_bleu:
            self.add_metric(SelfBleuMetric(field=field, bs=bs, **kwargs), DIRECTION_UNKNOWN)
        if enable_ref_bleu:
            self.add_metric(RefBleuMetric(field=field, bs=bs, **kwargs), DIRECTION_HIGHER_BETTER)

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
            (MetricBase): a metric object.
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
            (MetricBase): a metric object.
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

    def replace_target_classifier(self, clf):
        """Remove the original target classifier and add a new classifier."""
        del self._classifiers[self.get_target_classifier_name()]
        self.add_classifier(clf, set_target_clf=True)

    def measure_example(self, origin, paraphrase, data_record=None):
        """Compute the results of all metrics in the bundle for one pair of text.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record (dict): the data record.

        Returns:
            (dict): a dict with metric name as key.
        """
        ret = {}
        for name in self.get_metric_names():
            metric = self.get_metric(name)
            ret[name] = metric.measure_example(origin, paraphrase, data_record)
        for name in self.get_classifier_names():
            classifier = self.get_classifier(name)
            ret[name] = classifier.measure_example(
                origin, paraphrase, data_record)
        return ret

    def measure_batch(self, origin, paraphrase_list, data_record=None,):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.

        Returns:
            (list): a list containing dict of metrics for each paraphrase.
        """
        ret = [{} for i in range(len(paraphrase_list))]
        for name in self.get_metric_names():
            metric = self.get_metric(name)
            result = metric.measure_batch(origin, paraphrase_list, data_record)
            for i in range(len(paraphrase_list)):
                ret[i][name] = result[i]
        for name in self.get_classifier_names():
            classifier = self.get_classifier(name)
            result = classifier.measure_batch(origin, paraphrase_list, data_record)
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

        for data_record in tqdm.tqdm(results["data"]):
            data_record_tmp = dict([(k, v) for k, v in data_record.items()
                                    if "_paraphrases" not in k])

            # Run metrics on original text
            data_record["original_text_metrics"] = self.measure_example(
                data_record[self._field], data_record[self._field],
                data_record_tmp)

            # Run metrics on paraphrased text
            data_record["paraphrase_metrics"] = self.measure_batch(
                data_record[self._field], data_record[self._field + "_paraphrases"],
                data_record_tmp)

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
        aggregated_result["dataset_name"] = dataset_name
        aggregated_result["paraphrase_strategy_name"] = paraphrase_strategy_name
        aggregated_result["experiment_name"] = experiment_name

        return aggregated_result
