import json

import pandas as pd
import tqdm

from fibber import log
from fibber.datasets.preprocess_utils import download_raw_and_preprocess

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


def preprocess_ag_data(input_filename, output_filename):
    """preprocess raw AG's news csv to Fibber's JSON format."""
    logger.info("Start preprocessing data, and save at %s.", output_filename)
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


def download_and_preprocess_ag():
    """Download and preprocess AG's news dataset. """
    download_raw_and_preprocess(
        dataset_name="ag",
        download_list=["ag-raw-train", "ag-raw-test"],
        preprocess_fn=preprocess_ag_data,
        preprocess_input_output_list=[
            ("raw/train.csv", "train.json"),
            ("raw/test.csv", "test.json")])


if __name__ == "__main__":
    download_and_preprocess_ag()
