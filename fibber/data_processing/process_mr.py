import json

import tqdm

import pandas as pd

DATA_FOLDER = "data/mr/"


def main():
  with open(DATA_FOLDER + "/raw/rt-polarity.neg",
            encoding="utf-8", errors="ignore") as f:
    neg = f.readlines()

  with open(DATA_FOLDER + "/raw/rt-polarity.pos",
            encoding="utf-8", errors="ignore") as f:
    pos = f.readlines()

  train = {
      "label_mapping": ["neg", "pos"],
      "cased": False
  }

  test = {
      "label_mapping": ["neg", "pos"],
      "cased": False
  }

  trainlist = []
  testlist = []

  for id, item in enumerate(neg):
    if id % 10 == 0:
      trainlist.append({
          "label": 0,
          "s0": item.strip()})
    else:
      testlist.append({
          "label": 0,
          "s0": item.strip()})

  for id, item in enumerate(pos):
    if id % 10 == 0:
      trainlist.append({
          "label": 1,
          "s0": item.strip()})
    else:
      testlist.append({
          "label": 1,
          "s0": item.strip()})

  train["data"] = trainlist
  test["data"] = testlist

  with open(DATA_FOLDER + "/train.json", "w") as f:
    json.dump(train, f, indent=2)

  with open(DATA_FOLDER + "/test.json", "w") as f:
    json.dump(test, f, indent=2)


if __name__ == "__main__":
  main()
