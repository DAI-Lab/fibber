import copy
import hashlib
import json
import logging

import numpy as np
import tqdm

import torch
from transformers import BertTokenizer


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

  bins = [[] for i in dataset["label_mapping"]]

  subset = dict([(k, v) for k, v in dataset.items() if k != "data"])

  for idx, data in enumerate(dataset["data"]):
    label = data["label"]
    text = data["s0"]
    if "s1" in data:
      text += data["s1"]

    bins[label].append((idx, get_md5(text)))

  datalist = []
  for i in range(len(bins)):
    bins[i] = sorted(bins[i], key=lambda x: x[1])
    for j in range(n // len(bins)):
      datalist.append(copy.deepcopy(dataset["data"][idx]))

  subset["data"] = datalist

  return subset


class Dataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset, model_init, batch_size, exclude=-1,
               masked_lm=False, masked_lm_ratio=0.2, seed=0):

    self._buckets = [30, 50, 100, 200]
    self._max_len = self._buckets[-1]
    self._data = [[] for i in range(len(self._buckets))]

    self._batch_size = batch_size
    self._tokenizer = BertTokenizer.from_pretrained(model_init)

    self._seed = seed
    self._pad_tok_id = self._tokenizer.pad_token_id

    self._masked_lm = masked_lm
    self._masked_lm_ratio = masked_lm_ratio
    self._mask_tok_id = self._tokenizer.mask_token_id
    if masked_lm:
      assert mask_tok_id is not None

    counter = 0
    for item in tqdm.tqdm(dataset["data"]):
      y = item["label"]
      s0 = "[CLS] " + item["s0"]
      if "s1" in item:
        s1 = "[SEP] " + item["s1"]
      else:
        s1 = ""

      if y == exclude:
        continue

      counter += 1

      s0_ids = self._tokenizer.convert_tokens_to_ids(
          self._tokenizer.tokenize(s0))
      s1_ids = self._tokenizer.convert_tokens_to_ids(
          self._tokenizer.tokenize(s1))
      text_ids = (s0_ids + s1_ids)[:self._max_len]

      for bucket_id in range(len(self._buckets)):
        if self._buckets[bucket_id] >= len(text_ids):
          self._data[bucket_id].append((text_ids, y, len(s0_ids), len(s1_ids)))
          break

    logging.info("Load %d documents. with filter %d.", counter, exclude)
    self._bucket_prob = np.asarray([len(x) for x in self._data])
    self._bucket_prob = self._bucket_prob / np.sum(self._bucket_prob)

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      self._rng = np.random.RandomState(self._seed)
    else:
      self._rng = np.random.RandomState(worker_info.id + self._seed)

    while True:
      bucket_id = self._rng.choice(len(self._bucket_prob), p=self._bucket_prob)
      ids = self._rng.choice(len(self._data[bucket_id]), self._batch_size)

      texts, labels, l0s, l1s = zip(*[self._data[bucket_id][id] for id in ids])

      text_len = [len(x) for x in texts]
      max_text_len = max(text_len)

      # tok_list is copied by list add.
      texts = [x + [self._pad_tok_id] * (max_text_len - len(x)) for x in texts]
      masks = [[1] * x + [0] * (max_text_len - x) for x in text_len]
      tok_types = [([0] * l0 + [1] * l1 + [0] * (max_text_len - l0 - l1)
                    )[:max_text_len]
                   for (l0, l1) in zip(l0s, l1s)]

      texts = np.asarray(texts)
      masks = np.asarray(masks)
      labels = np.asarray(labels)
      tok_types = np.asarray(tok_types)

      if not self._masked_lm:
        yield (torch.tensor(texts), torch.tensor(masks),
               torch.tensor(tok_types), torch.tensor(labels))
      else:
        rand_t = self._rng.rand(self._batch_size, max_text_len)
        masked_pos = (rand_t < self._masked_lm_ratio) * masks
        masked_pos[:, 0] = 0
        lm_labels = (masked_pos * texts - 100 * (1 - masked_pos))
        rand_t = self._rng.rand(self._batch_size, max_text_len)
        filling = self._rng.randint(0, len(self._tokenizer),
                                    (self._batch_size, max_text_len))
        filling = ((rand_t < 0.8) * self._mask_tok_id
                   + (rand_t >= 0.8) * (rand_t < 0.9) * filling
                   + (rand_t >= 0.9) * texts)
        texts = masked_pos * filling + (1 - masked_pos) * texts
        yield (torch.tensor(texts), torch.tensor(masks), orch.tensor(tok_types),
               torch.tensor(labels), torch.tensor(lm_labels))
