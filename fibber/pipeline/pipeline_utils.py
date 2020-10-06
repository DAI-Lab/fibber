import datetime
import json

import tqdm

from .. import log
from ..measurement.measurement_utils import MeasurementBundle

logger = log.setup_custom_logger(__name__)


def measure_quality(dataset_name, trainset, testset, results,
                    paraphrase_field, output_filename, measurement_gpu):
    logger.info("Build measurement bundle.")
    measurement_bundle = MeasurementBundle(
        use_bert_clf_flip_pred=True,
        use_gpu_id=measurement_gpu, gpt2_gpu_id=measurement_gpu,
        bert_gpu_id=measurement_gpu, dataset_name=dataset_name,
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
