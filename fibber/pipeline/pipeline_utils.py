import datetime
import json
import os
import pandas as pd

import tqdm

from .. import log
from ..measurement.measurement_utils import MeasurementBundle

logger = log.setup_custom_logger(__name__)


def measure_quality(dataset_name, trainset, testset, results,
                    paraphrase_field, output_filename, gpt2_gpu,
                    bert_gpu, use_gpu):
    logger.info("Build measurement bundle.")
    measurement_bundle = MeasurementBundle(
        use_glove_semantic_similarity=False,

        use_bert_clf_flip_pred=True,
        use_gpu_id=use_gpu, gpt2_gpu_id=gpt2_gpu,
        bert_gpu_id=bert_gpu, dataset_name=dataset_name,
        trainset=trainset, testset=testset)

    last_output_save_time = -1
    logger.info("Start measuring bundle.")
    for data_record in tqdm.tqdm(results):
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

def aggregate_measurements(model_name, experiment_name, results, customize_metric, filename):
    if os.path.exists(filename):
        dataset_view = pd.read_csv(filename)
    else:
        dataset_view = pd.DataFrame()

    aggregate_result = pd.DataFrame()
    for data_record in results:
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
