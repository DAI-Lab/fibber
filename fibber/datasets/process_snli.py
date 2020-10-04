import json
import os

from .. import log
from ..download_utils import download_file, get_root_dir
from ..downloadable_resources import resources
import tqdm

logger = log.setup_custom_logger(__name__)


def process_data(input_filename, output_filename):
    logger.info("Start processing data, and save at %s.", output_filename)
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


def download_and_process_snli():
    root_dir = get_root_dir()
    dataset_dir = "datasets/snli/"

    download_file(filename=resources["snli-raw"]["filename"],
                  url=resources["snli-raw"]["url"],
                  md5_checksum=resources["snli-raw"]["md5"],
                  subdir=os.path.join(dataset_dir, "raw"), untar=False, unzip=True)

    process_data(os.path.join(root_dir, dataset_dir, "raw/snli_1.0/snli_1.0_train.jsonl"),
                 os.path.join(root_dir, dataset_dir, "train.json"))
    process_data(os.path.join(root_dir, dataset_dir, "raw/snli_1.0/snli_1.0_dev.jsonl"),
                 os.path.join(root_dir, dataset_dir, "dev.json"))
    process_data(os.path.join(root_dir, dataset_dir, "raw/snli_1.0/snli_1.0_test.jsonl"),
                 os.path.join(root_dir, dataset_dir, "test.json"))


if __name__ == "__main__":
    download_and_process_snli()
