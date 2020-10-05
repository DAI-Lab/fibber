import json
import os

import pandas as pd
import tqdm

from .. import log
from ..download_utils import download_file, get_root_dir
from ..downloadable_resources import resources

logger = log.setup_custom_logger(__name__)


REPLACE_TOKS = [
    ("#39;", "'"),
    ("#36;", "$"),
    ("&gt;", ">"),
    ("&lt;", "<"),
    ("\\$", "$"),
    ("quot;", "\""),
    ("\\", " "),
    ("#145;", "\""),
    ("#146;", "\""),
    ("#151;", "-")
]


def process_data(input_filename, output_filename):
    logger.info("Start processing data, and save at %s.", output_filename)
    df = pd.read_csv(input_filename, header=None)

    data = {
        "label_mapping": ["World", "Sports", "Business", "Sci/Tech"],
        "cased": True,
        "paraphrase_field": "text0",
    }

    datalist = []
    for item in tqdm.tqdm(df.iterrows(), total=len(df)):
        y, title, text = item[1]
        y -= 1
        for u, v in REPLACE_TOKS:
            title = title.replace(u, v)
            text = text.replace(u, v)

        datalist.append({
            "label": y,
            "text0": title + " . " + text
        })

    data["data"] = datalist
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)


def download_and_process_ag():
    root_dir = get_root_dir()
    dataset_dir = "datasets/ag/"

    download_file(filename="train.csv", url=resources["ag-raw-train"]["url"],
                  md5_checksum=resources["ag-raw-train"]["md5"],
                  subdir=os.path.join(dataset_dir, "raw"), untar=False)
    download_file(filename="test.csv", url=resources["ag-raw-test"]["url"],
                  md5_checksum=resources["ag-raw-test"]["md5"],
                  subdir=os.path.join(dataset_dir, "raw"), untar=False)

    process_data(os.path.join(root_dir, dataset_dir, "raw/train.csv"),
                 os.path.join(root_dir, dataset_dir, "train.json"))
    process_data(os.path.join(root_dir, dataset_dir, "raw/test.csv"),
                 os.path.join(root_dir, dataset_dir, "test.json"))


if __name__ == "__main__":
    download_and_process_ag()
