import glob
import json

import tqdm

import pandas as pd

DATA_FOLDER = "data/imdb/"


def process_data(input_folder, output_filename):
  data = {
      "label_mapping": ["neg", "pos"],
      "cased": True
  }

  datalist = []

  for filename in tqdm.tqdm(glob.glob(input_folder + "/neg/*.txt")):
    with open(filename, encoding="utf-8", errors="ignore") as f:
      text = "".join(f.readlines()).strip().replace("<br />", "")
      datalist.append({
          "label": 0,
          "s0": text
      })

  for filename in tqdm.tqdm(glob.glob(input_folder + "/pos/*.txt")):
    with open(filename, encoding="utf-8", errors="ignore") as f:
      text = "".join(f.readlines()).strip().replace("<br />", "")
      datalist.append({
          "label": 1,
          "s0": text
      })

  data["data"] = datalist
  with open(output_filename, "w") as f:
    json.dump(data, f, indent=2)


def main():
  process_data(DATA_FOLDER + "/raw/train/", DATA_FOLDER + "/train.json")
  process_data(DATA_FOLDER + "/raw/test/", DATA_FOLDER + "/test.json")


if __name__ == "__main__":
  main()
