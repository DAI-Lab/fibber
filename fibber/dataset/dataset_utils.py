import copy
import hashlib
import json
import os

import numpy as np
import torch
import tqdm
from transformers import BertTokenizer

from .. import log
from ..download_utils import get_root_dir

logger = log.setup_custom_logger(__name__)


def get_dataset(dataset_name):
    """Get datasets.

    Args:
        dataset_name: the name of the dataset.

    Returns:
        the training set and test set as a dictionary.
    """
    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "datasets")

    if dataset_name == "mnli" or dataset_name == "mnli_mis":
        train_filename = os.path.join(data_dir, "mnli/train.json")
        if dataset_name == "mnli":
            test_filename = os.path.join(data_dir, "mnli/dev_matched.json")
        else:
            test_filename = os.path.join(data_dir, "mnli/dev_mismatched.json")

    else:
        train_filename = os.path.join(data_dir, dataset_name, "train.json")
        test_filename = os.path.join(data_dir, dataset_name, "test.json")

    if not os.path.exists(train_filename) or not os.path.exists(test_filename):
        logger.error("%s dataset not found.", dataset_name)
        assert 0, ("Please use `python3 -m fibber.pipeline.download_datasets` "
                   "to download datasets.")

    with open(train_filename) as f:
        trainset = json.load(f)

    with open(test_filename) as f:
        testset = json.load(f)

    logger.info("%s training set has %d records.", dataset_name, len(trainset["data"]))
    logger.info("%s test set has %d records.", dataset_name, len(testset["data"]))
    return trainset, testset


def text_md5(x):
    m = hashlib.md5()
    m.update(x.encode('utf8'))
    return m.hexdigest()


def subsample_dataset(dataset, n):
    """Subsample n data from the dataset to n.

    Sample (n // k) examples from each category. Within each category, we pick examples with the
    lowest md5 hash value.

    Args:
        dataset: a dataset dictionary.
        n: the size of the subsampled dataset.

    Returns:
        a subsampled dataset as a dictionary.
    """
    if n > len(dataset["data"]):
        return copy.deepcopy(dataset)

    bins = [[] for i in dataset["label_mapping"]]

    subset = dict([(k, v) for k, v in dataset.items() if k != "data"])

    for idx, data in enumerate(dataset["data"]):
        label = data["label"]
        text = data["text0"]
        if "text1" in data:
            text += data["text1"]

        bins[label].append((idx, text_md5(text)))

    datalist = []
    for i in range(len(bins)):
        bins[i] = sorted(bins[i], key=lambda x: x[1])
        m = n // len(bins) + (1 if i < n % len(bins) else 0)
        for j in range(m):
            datalist.append(copy.deepcopy(dataset["data"][bins[i][j][0]]))

    subset["data"] = datalist

    return subset


def verify_dataset(data):
    """Verify if the dataset dictionary contains necessary fields.

    Args:
        data: a dataset dictionary.
    raises:
        assertion error when there are missing or incorrect fields.
    """
    assert "label_mapping" in data
    assert "cased" in data
    assert "paraphrase_field" in data
    assert data["paraphrase_field"] in ["text0", "text1"]

    num_labels = len(data["label_mapping"])
    counter = [0] * num_labels

    for data_record in tqdm.tqdm(data["data"]):
        assert "label" in data_record
        label = data_record["label"]
        assert 0 <= label < num_labels
        counter[label] += 1
        assert "text0" in data_record
        assert data["paraphrase_field"] in data_record

    for item in counter:
        assert item > 0, "empty class"


class DatasetForBert(torch.utils.data.IterableDataset):
    """Bert dataset wrapper for dataset dictionary. """

    def __init__(self, dataset, model_init, batch_size, exclude=-1,
                 masked_lm=False, masked_lm_ratio=0.2, seed=0):
        """Create dataset for a bert model.

        Args:
            dataset: a dataset dictionary.
            model_init: the pretrained model name. select from 'bert-base-cased',
                        'bert-base-uncased', 'bert-large-cased', and 'bert-large-uncased'.
            batch_size: the batch size in each step.
            exclude: exclude one category from the data.
                     Use -1 (default) to include all categories.
            masked_lm: whether to randomly replace words with mask tokens.
            masked_lm_ratio: the ratio of random masks. Ignored when masked_lm is False.
            seed: random seed.
        """

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

        counter = 0
        logger.info("DatasetForBert is processing data.")
        for item in tqdm.tqdm(dataset["data"]):
            y = item["label"]
            s0 = "[CLS] " + item["text0"]
            if "text1" in item:
                s1 = "[SEP] " + item["text1"]
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

        logger.info("Load %d documents. with filter %d.", counter, exclude)
        self._bucket_prob = np.asarray([len(x) for x in self._data])
        self._bucket_prob = self._bucket_prob / np.sum(self._bucket_prob)

    def __iter__(self):
        """Generate data.

        Returns:
            text: a tensor of size (batch_size, L).
            masks: a tensor of size (batch_size, L).
            tok_types: a tensor of size (batch_size, L).
            labels: a tensor of size (batch_size,).
            lm_labels: a tensor of size (batch_size, L). (when masked_lm is True)
        """
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
                yield (torch.tensor(texts), torch.tensor(masks), torch.tensor(tok_types),
                       torch.tensor(labels), torch.tensor(lm_labels))
