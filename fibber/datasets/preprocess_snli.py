import json

import tqdm

from fibber import log
from fibber.datasets.preprocess_utils import download_raw_and_preprocess

logger = log.setup_custom_logger(__name__)


def preprocess_snli_data(input_filename, output_filename):
    """preprocess SNLI dataset to Fibber's JSON format."""
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


def download_and_preprocess_snli():
    """Download and preprocess SNLI dataset."""
    download_raw_and_preprocess(
        dataset_name="snli",
        download_list=["snli-raw"],
        preprocess_fn=preprocess_snli_data,
        preprocess_input_output_list=[
            ("raw/snli_1.0/snli_1.0_train.jsonl", "train.json"),
            ("raw/snli_1.0/snli_1.0_dev.jsonl", "dev.json"),
            ("raw/snli_1.0/snli_1.0_test.jsonl", "test.json")])


if __name__ == "__main__":
    download_and_preprocess_snli()
