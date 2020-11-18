"""This module provides utility functions and classes to handle fibber's datasets.

* To load a dataset, use ``get_dataset`` function. For example, to load AG's news dataset, run::

    trainset, testset =  get_dataset("ag")

* The trainset and testset are both dicts. The dict looks like::

    {
      "label_mapping": [
        "World",
        "Sports",
        "Business",
        "Sci/Tech"
      ],
      "cased": true,
      "paraphrase_field": "text0",
      "data": [
        {
          "label": 1,
          "text0": "Boston won the NBA championship in 2008."
        },
        {
          "label": 3,
          "text0": "Apple releases its latest cell phone."
        },
        ...
      ]
    }

* To sub-sample 100 examples from training set, run::

    subsampled_dataset = subsample_dataset(trainset, 100)

* To convert a dataset dict to a ``torch.IterableDataset`` for BERT model, run::

    iterable_dataset = DatasetForBert(trainset, "bert-base-cased", batch_size=32);

For more details, see ``https://dai-lab.github.io/fibber/``
"""

import copy
import hashlib
import json
import os

import numpy as np
import torch
import tqdm
from transformers import BertTokenizerFast

from fibber import get_root_dir, log, resources
from fibber.datasets.downloadable_datasets import downloadable_dataset_urls
from fibber.download_utils import download_file

logger = log.setup_custom_logger(__name__)


def get_dataset(dataset_name):
    """Load dataset from fibber root directory.

    Users should make sure the data is downloaded to the ``datasets`` folder in fibber root
    dir (default: ``~/.fibber/datasets``). Otherwise, assertion error is raised.

    Args:
        dataset_name (str): the name of the dataset. See ``https://dai-lab.github.io/fibber/``
            for a full list of built-in datasets.

    Returns:
        (dict, dict): the function returns a tuple of two dict, representing the training set and
        test set respectively.
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
        assert 0, ("Please use `python3 -m fibber.datasets.download_datasets` "
                   "to download datasets.")

    with open(train_filename) as f:
        trainset = json.load(f)

    with open(test_filename) as f:
        testset = json.load(f)

    logger.info("%s training set has %d records.", dataset_name, len(trainset["data"]))
    logger.info("%s test set has %d records.", dataset_name, len(testset["data"]))
    return trainset, testset


def get_demo_dataset():
    """download demo dataset.

    Returns:
        (dict, dict): trainset and testset.
    """
    download_file(subdir="", **downloadable_dataset_urls["mr-demo"])

    data_dir = get_root_dir()
    data_dir = os.path.join(data_dir, "mr-demo")

    with open(os.path.join(data_dir, "train.json")) as f:
        trainset = json.load(f)
    with open(os.path.join(data_dir, "test.json")) as f:
        testset = json.load(f)

    logger.info("Demo training set has %d records.", len(trainset["data"]))
    logger.info("Demo test set has %d records.", len(testset["data"]))

    return trainset, testset


def text_md5(x):
    """Computes and returns the md5 hash of a str."""
    m = hashlib.md5()
    m.update(x.encode('utf8'))
    return m.hexdigest()


def subsample_dataset(dataset, n):
    """Sub-sample a dataset to `n` examples.

    Data is selected evenly and randomly from each category. Data in each category is sorted by
    its md5 hash value. The top ``(n // k)`` examples from each category are included in the
    sub-sampled dataset, where ``k`` is the number of categories.

    If ``n`` is not divisible by ``k``, one more data is sampled from the first ``(n % k)``
    categories.

    If the dataset has less than ``n`` examples, a copy of the original dataset will be returned.

    Args:
        dataset (dict): a dataset dict.
        n (int): the size of the sub-sampled dataset.

    Returns:
        (dict): a sub-sampled dataset as a dict.
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

    data_list = []
    for i in range(len(bins)):
        bins[i] = sorted(bins[i], key=lambda x: x[1])
        m = n // len(bins) + (1 if i < n % len(bins) else 0)
        for j in range(m):
            data_list.append(copy.deepcopy(dataset["data"][bins[i][j][0]]))

    subset["data"] = data_list

    return subset


def verify_dataset(dataset):
    """Verify if the dataset dict contains necessary fields.

    Assertion error is raised if there are missing or incorrect fields.

    Args:
        dataset (dict): a dataset dict.
    """
    assert "label_mapping" in dataset
    assert "cased" in dataset
    assert "paraphrase_field" in dataset
    assert dataset["paraphrase_field"] in ["text0", "text1"]

    num_labels = len(dataset["label_mapping"])
    counter = [0] * num_labels

    for data_record in tqdm.tqdm(dataset["data"]):
        assert "label" in data_record
        label = data_record["label"]
        assert 0 <= label < num_labels
        counter[label] += 1
        assert "text0" in data_record
        assert dataset["paraphrase_field"] in data_record

    for item in counter:
        assert item > 0, "empty class"


class DatasetForBert(torch.utils.data.IterableDataset):
    """Create a ``torch.IterableDataset`` for a BERT model.

    The module is an iterator that yields infinite batches from the dataset. To construct a
    batch, we randomly sample a few examples with similar length. Then we pad all selected
    examples to the same length ``L``. Then we construct a tuple of 4 or 5 tensors. All
    tensors are on CPU.

    Each example starts with ``[CLS]``, and ends with ``[SEP]``. If there are two parts in
    the input, the two parts are separated by ``[SEP]``.

    __iter__(self):

    Yields:
        A tuple of tensors.

        * The first tensor is an int tensor of size ``(batch_size, L)``, representing
          word ids. Each row of this tensor correspond to one example in the dataset.
          If ``masked_lm == True``, the tensor stores the masked text.
        * The second tensor is an int tensor of size ``(batch_size, L)``, representing
          the text length. Each entry is 1 if the corresponding position is text, and it
          is 0 if the position is padding.
        * The third tensor is an int tensor of size ``(batch_size, L)``, representing the
          token type. The token type is 0 if current position is in the first part of the
          input text. And it is 1 if current position is in the second part of the input.
          For padding positions, token type is 0.
        * The forth tensor an int tensor of size ``(batch_size,)``, representing the
          classification label.
        * (optional) If ``masked_lm == True``, the fifth tensor is a tensor of size
          ``(batch_size, L)``. Each entry in this tensor is either -100 if the position is
          not masked, or the correct word if the position is masked. Note that, a masked
          position is not always a ``[MASK]`` token in the first tensor. With 80%
          probability, it is a ``[MASK]``. With 10% probability, it is the original word.
          And with 10% probability, it is a random word.
    """

    def __init__(self, dataset, model_init, batch_size, exclude=-1,
                 masked_lm=False, masked_lm_ratio=0.2, seed=0):
        """Initialize.

        Args:
            dataset (dict): a dataset dict.
            model_init (str): the pre-trained model name. select from ``['bert-base-cased',
                'bert-base-uncased', 'bert-large-cased', and 'bert-large-uncased']``.
            batch_size (int): the batch size in each step.
            exclude (int): exclude one category from the data.
                Use -1 (default) to include all categories.
            masked_lm (bool): whether to randomly replace words with mask tokens.
            masked_lm_ratio (float): the ratio of random masks. Ignored when masked_lm is False.
            seed: random seed.
        """
        self._buckets = [30, 50, 100, 200]
        self._max_len = self._buckets[-1]
        self._data = [[] for i in range(len(self._buckets))]

        self._batch_size = batch_size
        self._tokenizer = BertTokenizerFast.from_pretrained(
            resources.get_transformers(model_init), do_lower_case="uncased" in model_init)

        self._seed = seed
        self._pad_tok_id = self._tokenizer.pad_token_id

        self._masked_lm = masked_lm
        self._masked_lm_ratio = masked_lm_ratio
        self._mask_tok_id = self._tokenizer.mask_token_id

        counter = 0
        logger.info("DatasetForBert is processing data.")
        for item in tqdm.tqdm(dataset["data"]):
            y = item["label"]
            s0 = "[CLS] " + item["text0"] + " [SEP]"
            if "text1" in item:
                s1 = item["text1"] + " [SEP]"
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
                filling = self._rng.randint(0, self._tokenizer.vocab_size,
                                            (self._batch_size, max_text_len))
                filling = ((rand_t < 0.8) * self._mask_tok_id
                           + (rand_t >= 0.8) * (rand_t < 0.9) * filling
                           + (rand_t >= 0.9) * texts)
                texts = masked_pos * filling + (1 - masked_pos) * texts
                yield (torch.tensor(texts), torch.tensor(masks), torch.tensor(tok_types),
                       torch.tensor(labels), torch.tensor(lm_labels))
