import json

import tqdm

import pandas as pd

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

DATA_FOLDER = "data/ag/"


def process_data(input_filename, output_filename):
  df = pd.read_csv(input_filename, header=None)

  data = {
      "label_mapping": ["World", "Sports", "Business", "Sci/Tech"],
      "cased": True
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
        "s0": title + " . " + text
    })

  data["data"] = datalist
  with open(output_filename, "w") as f:
    json.dump(data, f, indent=2)


def main():
  process_data(DATA_FOLDER + "/raw/train.csv", DATA_FOLDER + "/train.json")
  process_data(DATA_FOLDER + "/raw/test.csv", DATA_FOLDER + "/test.json")


if __name__ == "__main__":
  main()
