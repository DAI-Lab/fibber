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

    iterable_dataset = DatasetForTransformers(trainset, "bert-base-cased", batch_size=32);

For more details, see ``https://dai-lab.github.io/fibber/``
"""

import copy
import hashlib
import json
import os

import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer

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


def subsample_dataset(dataset, n, offset=0):
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
        offset (int): dataset offset.

    Returns:
        (dict): a sub-sampled dataset as a dict.
    """
    if n > len(dataset["data"]):
        return copy.deepcopy(dataset)

    bins = [[] for i in dataset["label_mapping"]]
    offset = offset // len(bins)

    subset = dict([(k, v) for k, v in dataset.items() if k != "data"])

    for idx, data in enumerate(dataset["data"]):
        label = data["label"]
        text = data["text0"]
        if "text1" in data:
            text += data["text1"]

        bins[label].append(idx)

    data_list = []
    for i in range(len(bins)):
        np.random.shuffle(bins[i])
        m = n // len(bins) + (1 if i < n % len(bins) else 0)
        for j in range(offset, min(offset + m, len(bins[i]))):
            data_list.append(copy.deepcopy(dataset["data"][bins[i][j]]))

    subset["data"] = data_list

    return subset


def verify_dataset(dataset):
    """Verify if the dataset dict contains necessary fields.

    Assertion error is raised if there are missing or incorrect fields.

    Args:
        dataset (dict): a dataset dict.
    """
    assert "label_mapping" in dataset

    num_labels = len(dataset["label_mapping"])
    counter = [0] * num_labels

    for data_record in tqdm.tqdm(dataset["data"]):
        assert "label" in data_record
        label = data_record["label"]
        assert 0 <= label < num_labels
        counter[label] += 1
        assert "text0" in data_record

    for item in counter:
        assert item > 0, "empty class"


def clip_sentence(dataset, model_init, max_len):
    """Inplace clipping sentences."""
    tokenizer = AutoTokenizer.from_pretrained(resources.get_transformers(model_init))
    logger.info("clipping the dataset to %d tokens.", max_len)
    for data_record in tqdm.tqdm(dataset["data"]):
        s0 = tokenizer.tokenize(data_record["text0"])
        s0 = s0[:max_len]
        data_record["text0"] = tokenizer.convert_tokens_to_string(s0)

        if "text1" in data_record:
            s1 = tokenizer.tokenize(data_record["text1"])
            s1 = s1[:(max_len - len(s0))]
            data_record["text1"] = tokenizer.convert_tokens_to_string(s1)


class DatasetForTransformers(torch.utils.data.IterableDataset):
    """Create a ``torch.IterableDataset`` for a BERT model.

    The module is an iterator that yields infinite batches from the dataset. To construct a
    batch, we randomly sample a few examples with similar length. Then we pad all selected
    examples to the same length ``L``. Then we construct a tuple of 4 or 5 tensors. All
    tensors are on CPU.

    Each example starts with ``[CLS]``, and ends with ``[SEP]``. If there are two parts in
    the input, the two parts are separated by ``[SEP]``.

    __iter__(self):

    Yields:
        A tuple of tensors (or list).

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
        * (optional) If ``masked_lm == True`` the fifth tensor is a tensor of size
          ``(batch_size, L)``. Each entry in this tensor is either -100 if the position is
          not masked, or the correct word if the position is masked. Note that, a masked
          position is not always a ``[MASK]`` token in the first tensor. With 80%
          probability, it is a ``[MASK]``. With 10% probability, it is the original word.
          And with 10% probability, it is a random word.
        * (optional) If ``autoregressive_lm == True`` the fifth tensor is a tensor of size
          ``(batch_size, L)``. Each entry in this tensor is either -100 if it's [CLS], [PAD].
        * (optional) if ``include_raw_text == True``, the last item is a list of str.
    """

    def __init__(self, dataset, model_init, batch_size, exclude=-1,
                 masked_lm=False, masked_lm_ratio=0.2, autoregressive_lm=False,
                 select_field=None):
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
            select_field (None or str): select one field. None to use all available fields.
        """
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(resources.get_transformers(model_init))

        self._pad_tok_id = self._tokenizer.pad_token_id

        self._masked_lm = masked_lm
        self._masked_lm_ratio = masked_lm_ratio
        self._mask_tok_id = self._tokenizer.mask_token_id

        if select_field is None:
            if "text1" not in dataset["data"]:
                self._field = "text0"
            else:
                self._field = "both"
        else:
            self._field = select_field

        self._autoregressive_lm = autoregressive_lm
        if self._autoregressive_lm and self._masked_lm:
            raise RuntimeError("masked_lm and autoregressive_lm are used at the same time.")

        self._data = dataset["data"]
        if exclude != -1:
            self._data = [item for item in self._data if item["label"] != exclude]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError("worker is not allowed.")

        while True:
            data_records = np.random.choice(self._data, self._batch_size)

            if self._field == "both":
                batch_input = self._tokenizer(
                    [item["text0"] for item in data_records],
                    [item["text1"] for item in data_records],
                    return_tensors="np",
                    padding=True)
            else:
                batch_input = self._tokenizer(
                    [item[self._field] for item in data_records],
                    return_tensors="np",
                    padding=True)

            texts = batch_input["input_ids"]
            masks = batch_input["attention_mask"]
            if "token_type_ids" in batch_input:
                tok_types = batch_input["token_type_ids"]
            else:
                tok_types = np.zeros_like(masks)
            labels = np.asarray([item["label"] for item in data_records])
            max_text_len = len(texts[0])

            if not self._masked_lm and not self._autoregressive_lm:
                ret = [torch.tensor(x) for x in [texts, masks, tok_types, labels]]
                yield tuple(ret)
            elif self._masked_lm:
                rand_t = np.random.rand(self._batch_size, max_text_len)
                masked_pos = (rand_t < self._masked_lm_ratio) * masks

                # No mask on cls and sep.
                masked_pos[:, 0] = 0
                masked_pos *= np.asarray(texts != self._tokenizer.sep_token_id, dtype="int")

                if np.sum(masked_pos) == 0:
                    continue
                lm_labels = (masked_pos * texts - 100 * (1 - masked_pos))
                rand_t = np.random.rand(self._batch_size, max_text_len)
                filling = np.random.randint(0, self._tokenizer.vocab_size,
                                            (self._batch_size, max_text_len))
                filling = ((rand_t < 0.8) * self._mask_tok_id
                           + (rand_t >= 0.8) * (rand_t < 0.9) * filling
                           + (rand_t >= 0.9) * texts)
                texts = masked_pos * filling + (1 - masked_pos) * texts

                ret = [torch.tensor(x) for x in [texts, masks, tok_types, labels, lm_labels]]
                yield tuple(ret)
            elif self._autoregressive_lm:
                lm_labels = texts * masks - 100 * (1 - masks)
                # shift lm labels
                lm_labels[:, :-1] = lm_labels[:, 1:]
                lm_labels[:, -1] = -100

                ret = [torch.tensor(x) for x in [texts, masks, tok_types, labels, lm_labels]]
                yield tuple(ret)
            else:
                raise RuntimeError("unexpected branch.")
