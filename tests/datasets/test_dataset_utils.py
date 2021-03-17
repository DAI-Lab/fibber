import numpy as np
import pytest
import torch

from fibber.datasets.dataset_utils import DatasetForBert, KeywordsExtractor


@pytest.fixture()
def mock_dataset():
    return {
        "cased": False,
        "paraphrase_field": "text0",
        "label_mapping": ["negative", "positive"],
        "data": [
            {
                "label": 0,
                "text0": "Saturday is the day after Friday."
            }
        ]
    }


def test_keywords_extractor():
    extractor = KeywordsExtractor()

    text = "Saturday is the day after Friday."
    reference = ["saturday", "friday", "day"]
    reference_keep_order = ["saturday", "day", "friday"]

    result = extractor.extract_keywords(text, keep_order=False)
    print(result, reference)
    assert len(result) == len(reference) and all([x == y for x, y in zip(result, reference)])

    result = extractor.extract_keywords(text, keep_order=True)
    print(result, reference_keep_order)
    assert (len(result) == len(reference) and
            all([x == y for x, y in zip(result, reference_keep_order)]))


def test_dataset_with_kw_and_padding(mock_dataset):
    dataset = DatasetForBert(
        mock_dataset, "bert-base-uncased", 1, masked_lm=False,
        include_raw_text=False, dynamic_masked_lm=True, kw_and_padding=True, num_keywords=3)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=1)
    dataloader_iter = iter(dataloader)

    texts, masks, tok_types, labels, lm_labels = next(dataloader_iter)

    texts_size = texts.size()
    assert texts_size[0] == 1 and texts_size[1] == 16

    reference = [101, 0, 5095, 1010, 2154, 1010, 5958, 102]
    reference_str = "[CLS] [PAD] saturday, day, friday [SEP]"

    assert all([x == y for x, y in zip(texts[0, :len(reference)], reference)])

    assert dataset._tokenizer.decode(texts[0, :len(reference)].numpy()) == reference_str

    assert np.all(masks.numpy())

    assert np.all(tok_types[0, :len(reference)].numpy() == 0)

    assert np.all(tok_types[0, len(reference):].numpy() == 1)

    assert labels[0] == 0
