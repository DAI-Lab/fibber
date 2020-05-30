import json

import tqdm

import pandas as pd

REPLACE_TOKS = [
    ("\\\"", "\""),
    ("\\n", " ")
]

DATA_FOLDER = "data/yelp/"


def process_data(input_filename, output_filename):
  df = pd.read_csv(input_filename, header=None)

  data = {
      "label_mapping": ["neg", "pos"],
      "cased": True
  }

  datalist = []
  for item in tqdm.tqdm(df.iterrows(), total=len(df)):
    y, text = item[1]
    y -= 1
    for u, v in REPLACE_TOKS:
      text = text.replace(u, v)

    datalist.append({
        "label": y,
        "s0": text
    })

  data["data"] = datalist
  with open(output_filename, "w") as f:
    json.dump(data, f, indent=2)


def main():
  process_data(DATA_FOLDER + "/raw/train.csv", DATA_FOLDER + "/train.json")
  process_data(DATA_FOLDER + "/raw/test.csv", DATA_FOLDER + "/test.json")


if __name__ == "__main__":
  main()
