import glob
import json
import os

import tqdm

from .. import log
from ..download_utils import download_file, get_root_dir
from ..downloadable_resources import resources

logger = log.setup_custom_logger(__name__)


def process_data(input_folder, output_filename):
    logger.info("Start processing data, and save at %s.", output_filename)
    data = {
        "label_mapping": ["negative", "positive"],
        "cased": True,
        "paraphrase_field": "text0",
    }

    datalist = []

    for filename in tqdm.tqdm(glob.glob(input_folder + "/neg/*.txt")):
        with open(filename, encoding="utf-8", errors="ignore") as f:
            text = "".join(f.readlines()).strip().replace("<br />", "")
            datalist.append({
                "label": 0,
                "text0": text
            })

    for filename in tqdm.tqdm(glob.glob(input_folder + "/pos/*.txt")):
        with open(filename, encoding="utf-8", errors="ignore") as f:
            text = "".join(f.readlines()).strip().replace("<br />", "")
            datalist.append({
                "label": 1,
                "text0": text
            })

    data["data"] = datalist
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=2)


def download_and_process_imdb():
    root_dir = get_root_dir()
    dataset_dir = "datasets/imdb/"

    download_file(filename=resources["imdb-raw"]["filename"],
                  url=resources["imdb-raw"]["url"],
                  md5_checksum=resources["imdb-raw"]["md5"],
                  subdir=os.path.join(dataset_dir, "raw"), untar=True)

    process_data(os.path.join(root_dir, dataset_dir, "raw/aclImdb/train"),
                 os.path.join(root_dir, dataset_dir, "train.json"))
    process_data(os.path.join(root_dir, dataset_dir, "raw/aclImdb/test"),
                 os.path.join(root_dir, dataset_dir, "test.json"))


if __name__ == "__main__":
    download_and_process_imdb()
