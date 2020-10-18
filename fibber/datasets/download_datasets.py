"""This module downloads fibber datasets.

To download preprocessed datasets (Recommended), run::

    python -m fibber.datasets.download_datasets

To download datasets from their original sources, and preprocess them locally::

    python -m fibber.datasets.download_datasets --process_raw 1
"""


import argparse
import glob
import json
import os

from fibber import get_root_dir, log
from fibber.datasets import (
    preprocess_ag, preprocess_imdb, preprocess_mnli, preprocess_mr, preprocess_snli,
    preprocess_yelp)
from fibber.datasets.dataset_utils import verify_dataset
from fibber.datasets.downloadable_datasets import downloadable_dataset_urls
from fibber.download_utils import download_file

logger = log.setup_custom_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--process_raw", choices=["0", "1"], default="0",
                    help="Use 1 to download and process raw data on this machine. "
                    "Use 0 (Default) to download processed data.")
parser.add_argument("--verify", choices=["0", "1"], default="1",
                    help="Verify each json in each datasets have proper attributes.")

DATASET_PREPROCESS_FN = {
    "ag": preprocess_ag.download_and_preprocess_ag,
    "imdb": preprocess_imdb.download_and_preprocess_imdb,
    "mnli": preprocess_mnli.download_and_preprocess_mnli,
    "mr": preprocess_mr.download_and_preprocess_mr,
    "snli": preprocess_snli.download_and_preprocess_snli,
    "yelp": preprocess_yelp.download_and_preprocess_yelp
}

if __name__ == "__main__":
    FLAGS = parser.parse_args()

    if FLAGS.process_raw == "1":
        for name, processing_func in DATASET_PREPROCESS_FN.items():
            logger.info("Start download and process %s.", name)
            processing_func()

    else:
        download_file(subdir="", **downloadable_dataset_urls["processed-datasets"])

    if FLAGS.verify == "1":
        root_dir = get_root_dir()
        datasets_dir = os.path.join(root_dir, "datasets")
        dataset_json_list = sorted(glob.glob(datasets_dir + "/*/*.json"))
        for json_filename in dataset_json_list:
            logger.info("Verify %s.", json_filename)
            with open(json_filename) as f:
                data = json.load(f)

            verify_dataset(data)
