import glob
import json

import tqdm

from fibber import log
from fibber.datasets.preprocess_utils import download_raw_and_preprocess

logger = log.setup_custom_logger(__name__)


def preprocess_imdb_data(input_folder, output_filename):
    """preprocess raw IMDB dataset folder to Fibber's JSON format."""

    logger.info("Start preprocessing data, and save at %s.", output_filename)
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


def download_and_preprocess_imdb():
    """Download and preprocess IMDB sentiment classification dataset. """
    download_raw_and_preprocess(
        dataset_name="imdb",
        download_list=["imdb-raw"],
        preprocess_fn=preprocess_imdb_data,
        preprocess_input_output_list=[
            ("raw/aclImdb/train", "train.json"),
            ("raw/aclImdb/test", "test.json")])


if __name__ == "__main__":
    download_and_preprocess_imdb()
