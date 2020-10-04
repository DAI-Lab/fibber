import json
import os

import pandas as pd
import tqdm

from .. import log
from ..download_utils import download_file, get_root_dir
from ..downloadable_resources import resources

logger = log.setup_custom_logger(__name__)

REPLACE_TOKS = [
    ("\\\"", "\""),
    ("\\n", " ")
]


def process_data(input_filename, output_filename):
    logger.info("Start processing data, and save at %s.", output_filename)

    df = pd.read_csv(input_filename, header=None)

    data = {
        "label_mapping": ["negative", "positive"],
        "cased": True,
        "paraphrase_field": "text0",
    }

    datalist = []
    for item in tqdm.tqdm(df.iterrows(), total=len(df)):
        y, text = item[1]
        y -= 1
        for u, v in REPLACE_TOKS:
            text = text.replace(u, v)

        datalist.append({
            "label": y,
            "text0": text
        })

    data["data"] = datalist
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)


def download_and_process_yelp():
    root_dir = get_root_dir()
    dataset_dir = "datasets/yelp/"

    download_file(filename=resources["yelp-raw"]["filename"],
                  url=resources["yelp-raw"]["url"],
                  md5_checksum=resources["yelp-raw"]["md5"],
                  subdir=os.path.join(dataset_dir, "raw"), untar=True)

    process_data(os.path.join(root_dir, dataset_dir, "raw/yelp_review_polarity_csv/train.csv"),
                 os.path.join(root_dir, dataset_dir, "train.json"))
    process_data(os.path.join(root_dir, dataset_dir, "raw/yelp_review_polarity_csv/test.csv"),
                 os.path.join(root_dir, dataset_dir, "tesst.json"))


if __name__ == "__main__":
    download_and_process_yelp()
