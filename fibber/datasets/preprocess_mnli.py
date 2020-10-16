import json

import tqdm

from fibber import log
from fibber.datasets.preprocess_utils import download_raw_and_preprocess

logger = log.setup_custom_logger(__name__)


def preprocess_mnli_data(input_filename, output_filename):
    """preprocess raw MNLI dataset to Fibber's JSON format."""
    logger.info("Start preprocessing data, and save at %s.", output_filename)

    with open(input_filename) as f:
        df = f.readlines()
    df = [json.loads(line) for line in df]

    mapping = {
        "neutral": 0,
        "entailment": 1,
        "contradiction": 2
    }
    data = {
        "label_mapping": ["neutral", "entailment", "contradiction"],
        "cased": True,
        "paraphrase_field": "text1",
    }

    datalist = []
    for item in tqdm.tqdm(df):
        if item["gold_label"] == '-':
            continue
        y = mapping[item["gold_label"]]
        datalist.append({
            "label": y,
            "text0": item["sentence1"],
            "text1": item["sentence2"]
        })

    data["data"] = datalist
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)


def download_and_preprocess_mnli():
    """Download and preprocess MNLI dataset."""
    download_raw_and_preprocess(
        dataset_name="mnli",
        download_list=["mnli-raw"],
        preprocess_fn=preprocess_mnli_data,
        preprocess_input_output_list=[
            ("raw/multinli_1.0/multinli_1.0_train.jsonl", "train.json"),
            ("raw/multinli_1.0/multinli_1.0_dev_matched.jsonl", "dev_matched.json"),
            ("raw/multinli_1.0/multinli_1.0_dev_mismatched.jsonl", "dev_mismatched.json")])


if __name__ == "__main__":
    download_and_preprocess_mnli()
