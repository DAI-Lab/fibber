import datetime
import json
import os

import pandas as pd
import tqdm

from .. import log
from .bert_clf_flip_pred import BertClfFlipPred
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
                 use_bert_clf_flip_pred=False,
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
        if use_bert_clf_flip_pred:
            self._measurements.append(BertClfFlipPred(**kargs))
        self._measurements += customized_measurements

    def _evaluate(self, origin, paraphrase):
        ret = {}
        for measurement in self._measurements:
            ret[str(measurement)] = measurement(origin, paraphrase)
        return ret

    def __call__(self, origin, paraphrase):
        if isinstance(origin, str):
            assert isinstance(paraphrase, str)
            return self._evaluate(origin, paraphrase)

        assert len(origin) == len(paraphrase)
        ret = []
        for u, v in zip(origin, paraphrase):
            ret.append(self._evaluate(u, v))
        return ret


def measure_quality(dataset_name, trainset, testset, results,
                    paraphrase_field, output_filename, gpt2_gpu,
                    bert_gpu, use_gpu):
    logger.info("Build measurement bundle.")
    measurement_bundle = MeasurementBundle(
        use_bert_clf_flip_pred=True,
        use_gpu_id=use_gpu, gpt2_gpu_id=gpt2_gpu,
        bert_gpu_id=bert_gpu, dataset_name=dataset_name,
        trainset=trainset, testset=testset)

    last_output_save_time = -1
    logger.info("Start measuring bundle.")
    for data_record in tqdm.tqdm(results["data"]):
        paraphrase_measurement_list = []
        for paraphrase in data_record[paraphrase_field + "_paraphrases"]:
            paraphrase_measurement_list.append(
                measurement_bundle(data_record[paraphrase_field], paraphrase))

        data_record["paraphrase_measurement"] = paraphrase_measurement_list

        # save tmp output every 30 seconds
        if datetime.datetime.now().timestamp() - last_output_save_time > 30:
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
            datetime.datetime.now().timestamp()

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_measurements(model_name, experiment_name, results, customize_metric,
                           filename):
    if os.path.exists(filename):
        dataset_view = pd.read_csv(filename)
    else:
        dataset_view = pd.DataFrame()

    aggregate_result = pd.DataFrame()
    for data_record in results["data"]:
        agg_t = dict(pd.DataFrame(data_record["paraphrase_measurement"]).mean())
        agg_t = dict([(k + "_Avg", v) for k, v in agg_t.items()])
        agg_t["ParaphrasesPerExample"] = len(data_record["paraphrase_measurement"])

        for name, fn in customize_metric.items():
            agg_t[name] = fn(data_record["paraphrase_measurement"])

        aggregate_result = aggregate_result.append(agg_t, ignore_index=True)

    aggregate_result = dict(aggregate_result.mean())
    # hack column order by adding 0
    aggregate_result["0_model_name"] = model_name
    aggregate_result["0_experiment_name"] = experiment_name

    dataset_view = dataset_view.append(aggregate_result, ignore_index=True)
    dataset_view.to_csv(filename, index=None)
    return dataset_view
