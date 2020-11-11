import json
import re

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


def preprocess_ag_data(input_filename, output_filename,
                       include_title=True, include_author_media=True):
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
            if not include_author_media:
                m = re.search(r"^[^\(\)]{0,50}?(\([^\(\)]{0,20}?\))? (-|--) ", text)
                if m is not None and len(m.group(0)) < len(text):
                    text = text[len(m.group(0)):]

        datalist.append({
            "label": y,
            "text0": (title + " . " + text) if include_title else text
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

    download_raw_and_preprocess(
        dataset_name="ag_no_title",
        download_list=["ag-raw-train", "ag-raw-test"],
        preprocess_fn=lambda inp, out:
            preprocess_ag_data(inp, out, include_title=False, include_author_media=False),
        preprocess_input_output_list=[
            ("raw/train.csv", "train.json"),
            ("raw/test.csv", "test.json")])


if __name__ == "__main__":
    download_and_preprocess_ag()
