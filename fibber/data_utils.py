import copy
import hashlib
import json


def load_data(datafolder, dataset):
  if dataset == "mnli" or dataset == "mnli_mis":
    train_filename = datafolder + "/mnli/train.json"
    if dataset == "mnli":
      test_filename = datafolder + "/mnli/dev_matched.json"
    else:
      test_filename = datafolder + "/mnli/dev_mismatched.json"

  else:
    train_filename = datafolder + "/" + dataset + "/train.json"
    test_filename = datafolder + "/" + dataset + "/test.json"

  with open(train_filename) as f:
    trainset = json.load(f)

  with open(test_filename) as f:
    testset = json.load(f)

  return trainset, testset


def get_md5(x):
  m = hashlib.md5()
  m.update(x.encode('utf8'))
  return m.hexdigest()


def subsample_data(dataset, n):
  if n > len(dataset["data"]):
    return copy.deepcopy(dataset)

  bins=[[] for i in dataset["label_mapping"]]

  subset=dict([(k, v) for k, v in dataset.items() if k != "data"])

  for idx, data in enumerate(dataset["data"]):
    label=data["label"]
    text=data["s0"]
    if "s1" in data:
      text += data["s1"]

    bins[label].append((idx, get_md5(text)))

  datalist = []
  for i in range(len(bins)):
    bins[i]=sorted(bins[i], key=lambda x: x[1])
    for j in range(n // len(bins)):
      datalist.append(copy.deepcopy(dataset["data"][idx]))

  subset["data"] = datalist

  return subset
