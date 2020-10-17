import json

import pandas as pd
import tqdm

from fibber import log
from fibber.datasets.preprocess_utils import download_raw_and_preprocess

logger = log.setup_custom_logger(__name__)

REPLACE_TOKS = [
    ("\\\"", "\""),
    ("\\n", " ")
]


def preprocess_yelp_data(input_filename, output_filename):
    """preprocess Yelp dataset to Fibber's JSON format."""
    logger.info("Start preprocessing data, and save at %s.", output_filename)

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


def download_and_preprocess_yelp():
    """Download and preprocess Yelp dataset."""
    download_raw_and_preprocess(
        dataset_name="yelp",
        download_list=["yelp-raw"],
        preprocess_fn=preprocess_yelp_data,
        preprocess_input_output_list=[
            ("raw/yelp_review_polarity_csv/train.csv", "train.json"),
            ("raw/yelp_review_polarity_csv/test.csv", "test.json")])


if __name__ == "__main__":
    download_and_preprocess_yelp()
