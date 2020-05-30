import json

import tqdm

import pandas as pd

DATA_FOLDER = "data/mnli/"


def process_data(input_filename, output_filename):
  with open(input_filename) as f:
    df = f.readlines()
  df = [json.loads(line) for line in df]

  mapping = {
      "neutral": 0,
      "entailment": 1,
      "contradiction": 2
  }
  data = {
      "label_mapping": ["neutral", "entailment", "contradiction"],
      "cased": True
  }

  datalist = []
  for item in tqdm.tqdm(df):
    if item["gold_label"] == '-':
      continue
    y = mapping[item["gold_label"]]
    datalist.append({
        "label": y,
        "s0": item["sentence1"],
        "s1": item["sentence2"]
    })

  data["data"] = datalist
  with open(output_filename, "w") as f:
    json.dump(data, f, indent=2)


def main():
  process_data(DATA_FOLDER + "/raw/train.jsonl", DATA_FOLDER + "/train.json")
  process_data(DATA_FOLDER + "/raw/dev_matched.jsonl",
               DATA_FOLDER + "/dev_matched.json")
  process_data(DATA_FOLDER + "/raw/dev_mismatched.jsonl",
               DATA_FOLDER + "/dev_mismatched.json")


if __name__ == "__main__":
  main()
