import copy
import hashlib
import json
import os

import tqdm

from .. import log
from ..download_utils import get_root_dir

logger = log.setup_custom_logger(__name__)


def get_dataset(dataset_name):
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "datasets")

    if dataset_name == "mnli" or dataset_name == "mnli_mis":
        train_filename = os.path.join(data_dir, "mnli/train.json")
        if dataset_name == "mnli":
            test_filename = os.path.join(data_dir, "mnli/dev_matched.json")
        else:
            test_filename = os.path.join(data_dir, "mnli/dev_mismatched.json")

    else:
        train_filename = os.path.join(data_dir, dataset_name, "train.json")
        test_filename = os.path.join(data_dir, dataset_name, "test.json")

    if not os.path.exists(train_filename) or not os.path.exists(test_filename):
        logger.error("%s dataset not found.", dataset_name)
        assert 0, ("Please use `python3 -m fibber.pipeline.download_datasets` "
                   "to download datasets.")

    with open(train_filename) as f:
        trainset = json.load(f)

    with open(test_filename) as f:
        testset = json.load(f)

    logger.info("%s training set has %d records.", dataset_name, len(trainset["data"]))
    logger.info("%s test set has %d records.", dataset_name, len(testset["data"]))
    return trainset, testset


def text_md5(x):
    m = hashlib.md5()
    m.update(x.encode('utf8'))
    return m.hexdigest()


def subsample_dataset(dataset, n):
    if n > len(dataset["data"]):
        return copy.deepcopy(dataset)

    bins = [[] for i in dataset["label_mapping"]]

    subset = dict([(k, v) for k, v in dataset.items() if k != "data"])

    for idx, data in enumerate(dataset["data"]):
        label = data["label"]
        text = data["text0"]
        if "text1" in data:
            text += data["text1"]

        bins[label].append((idx, text_md5(text)))

    datalist = []
    for i in range(len(bins)):
        bins[i] = sorted(bins[i], key=lambda x: x[1])
        m = n // len(bins) + (1 if i < n % len(bins) else 0)
        for j in range(m):
            datalist.append(copy.deepcopy(dataset["data"][bins[i][j][0]]))

    subset["data"] = datalist

    return subset


def verify_dataset(data):
    assert "label_mapping" in data
    assert "cased" in data
    assert "paraphrase_field" in data
    assert data["paraphrase_field"] in ["text0", "text1"]

    num_labels = len(data["label_mapping"])
    counter = [0] * num_labels

    for data_record in tqdm.tqdm(data["data"]):
        assert "label" in data_record
        label = data_record["label"]
        assert 0 <= label < num_labels
        counter[label] += 1
        assert "text0" in data_record
        assert data["paraphrase_field"] in data_record

    for item in counter:
        assert item > 0, "empty class"
