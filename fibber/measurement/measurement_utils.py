import datetime
import json

import numpy as np
import pandas as pd
import tqdm

from .. import log
from .bert_clf_prediction import BertClfPrediction
from .editing_distance import EditingDistance
from .glove_semantic_similarity import GloVeSemanticSimilarity
from .gpt2_grammar_quality import GPT2GrammarQuality
from .measurement_base import MeasurementBase
from .use_semantic_similarity import USESemanticSimilarity

logger = log.setup_custom_logger(__name__)


class MeasurementBundle(object):
    """docstring for MeasurementBundle."""

    def __init__(self,
                 use_editing_distance=True,
                 use_use_semantic_simialrity=True,
                 use_glove_semantic_similarity=True,
                 use_gpt2_grammar_quality=True,
                 use_bert_clf_prediction=False,
                 customized_measurements=[],
                 **kargs):
        super(MeasurementBundle, self).__init__()

        assert isinstance(customized_measurements, list)
        for item in customized_measurements:
            assert isinstance(item, MeasurementBase)

        self._measurements = []
        if use_editing_distance:
            self._measurements.append(EditingDistance(**kargs))
        if use_use_semantic_simialrity:
            self._measurements.append(USESemanticSimilarity(**kargs))
        if use_glove_semantic_similarity:
            self._measurements.append(GloVeSemanticSimilarity(**kargs))
        if use_gpt2_grammar_quality:
            self._measurements.append(GPT2GrammarQuality(**kargs))
        if use_bert_clf_prediction:
            self._measurements.append(BertClfPrediction(**kargs))
        self._measurements += customized_measurements

    def _evaluate(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        ret = {}
        for measurement in self._measurements:
            ret[str(measurement)] = measurement(origin, paraphrase, data_record, paraphrase_field)
        return ret

    def __call__(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        if isinstance(origin, str):
            assert isinstance(paraphrase, str)
            return self._evaluate(origin, paraphrase, data_record, paraphrase_field)

        assert len(origin) == len(paraphrase)
        ret = []
        for u, v in zip(origin, paraphrase):
            ret.append(self._evaluate(u, v))
        return ret


def measure_quality(dataset_name, trainset, testset, results,
                    paraphrase_field, output_filename, gpt2_gpu,
                    bert_gpu, use_gpu, **kargs):
    logger.info("Build measurement bundle.")
    measurement_bundle = MeasurementBundle(
        use_bert_clf_prediction=True,
        use_gpu_id=use_gpu, gpt2_gpu_id=gpt2_gpu,
        bert_gpu_id=bert_gpu, dataset_name=dataset_name,
        trainset=trainset, testset=testset, **kargs)

    last_output_save_time = -1
    logger.info("Start measuring bundle.")
    for data_record in tqdm.tqdm(results["data"]):
        paraphrase_measurement_list = []
        data_record_tmp = dict([(k, v) for k, v in data_record.items() if "_paraphrases" not in k])
        for paraphrase in data_record[paraphrase_field + "_paraphrases"]:
            paraphrase_measurement_list.append(
                measurement_bundle(data_record[paraphrase_field], paraphrase, data_record_tmp,
                                   paraphrase_field))

        data_record["paraphrase_measurements"] = paraphrase_measurement_list

        # save tmp output every 30 seconds
        if datetime.datetime.now().timestamp() - last_output_save_time > 30:
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
            datetime.datetime.now().timestamp()

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    return results


MAJORITY_AGG = "majority"
MEAN_AGG = "mean"
STD_AGG = "std"

SPECIAL_METRIC_AGGREGATION = {
    "BertClfPrediction": []
}
DEFAULT_AGGREGATION = [MEAN_AGG, STD_AGG]


def mean_aggregation_fn(x):
    return float(np.mean(x))


def std_aggregation_fn(x):
    return float(np.std(x))


def majority_aggregation_fn(x):
    (values, counts) = np.unique(x, return_counts=True)
    return int(np.argmax(counts))


AGGREGATION_NAME_TO_FN = {
    MAJORITY_AGG: majority_aggregation_fn,
    MEAN_AGG: mean_aggregation_fn,
    STD_AGG: std_aggregation_fn
}


def aggregate_measurements(model_name, experiment_name, results, customize_metric):
    aggregated_result = pd.DataFrame()
    for data_record in results["data"]:
        aggregate_result_tmp = {}
        metric_df = pd.DataFrame(data_record["paraphrase_measurements"])
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

        aggregate_result_tmp["ParaphrasesPerExample"] = len(data_record["paraphrase_measurements"])

        for name, fn in customize_metric.items():
            aggregate_result_tmp[name] = fn(data_record)

        aggregated_result = aggregated_result.append(aggregate_result_tmp, ignore_index=True)

    aggregated_result = dict(aggregated_result.mean())
    # hack column order by adding 0
    aggregated_result["1_model_name"] = model_name
    aggregated_result["2_experiment_name"] = experiment_name

    return aggregated_result
