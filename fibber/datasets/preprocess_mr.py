import json
import os

from fibber import log
from fibber.datasets.downloadable_datasets import downloadable_dataset_urls
from fibber.download_utils import download_file, get_root_dir

logger = log.setup_custom_logger(__name__)


def download_and_preprocess_mr():
    """preprocess raw movie review dataset to Fibber's JSON format."""
    root_dir = get_root_dir()
    dataset_dir = "datasets/mr/"

    download_file(subdir=os.path.join(dataset_dir, "raw"),
                  **downloadable_dataset_urls["mr-raw"])

    logger.info("Start processing data.")

    with open(os.path.join(root_dir, dataset_dir, "raw/rt-polaritydata/rt-polarity.neg"),
              encoding="utf-8", errors="ignore") as f:
        neg = f.readlines()

    with open(os.path.join(root_dir, dataset_dir, "raw/rt-polaritydata/rt-polarity.pos"),
              encoding="utf-8", errors="ignore") as f:
        pos = f.readlines()

    train = {
        "label_mapping": ["negative", "positive"],
        "cased": False,
        "paraphrase_field": "text0",
    }

    test = {
        "label_mapping": ["negative", "positive"],
        "cased": False,
        "paraphrase_field": "text0",
    }

    trainlist = []
    testlist = []

    for id, item in enumerate(neg):
        if id % 10 == 0:
            testlist.append({
                "label": 0,
                "text0": item.strip()})
        else:
            trainlist.append({
                "label": 0,
                "text0": item.strip()})

    for id, item in enumerate(pos):
        if id % 10 == 0:
            testlist.append({
                "label": 1,
                "text0": item.strip()})
        else:
            trainlist.append({
                "label": 1,
                "text0": item.strip()})

    train["data"] = trainlist
    test["data"] = testlist

    with open(os.path.join(root_dir, dataset_dir, "train.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(root_dir, dataset_dir, "test.json"), "w") as f:
        json.dump(test, f, indent=2)


if __name__ == "__main__":
    download_and_preprocess_mr()
