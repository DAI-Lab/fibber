import numpy as np
import pytest
import torch

from fibber.datasets.dataset_utils import get_demo_dataset
from fibber.metrics.bert_classifier import BertClassifier
from fibber.resources.resource_utils import get_bert_clf_demo


@pytest.fixture()
def gpu_id():
    if torch.cuda.device_count() > 0:
        return 0
    return -1


@pytest.fixture()
def bert_classifier_on_demo(gpu_id):
    get_bert_clf_demo()
    trainset, testset = get_demo_dataset()
    bert_classifier = BertClassifier(
        "demo", trainset, testset, bert_gpu_id=gpu_id, bert_clf_steps=5000)
    return bert_classifier


@pytest.mark.slow
def test_bert_classifier(bert_classifier_on_demo):
    assert isinstance(bert_classifier_on_demo, BertClassifier)

    io_pairs = [
        ("This is a bad movie", 0),
        ("This is a good movie", 1)
    ]

    for x, y in io_pairs:
        dist = bert_classifier_on_demo.predict_dist_example(None, x)
        assert len(dist) == 2
        assert np.argmax(dist) == y

        pred = bert_classifier_on_demo.predict_example(None, x)
        assert pred == y

        pred = bert_classifier_on_demo.measure_example(None, x)
        assert pred == y

    batched_io_pairs = [
        (["This is a bad movie.",
          "This is a good movie. I want to buy a ticket."],
         [0, 1])
    ]

    for x, y in batched_io_pairs:
        dist = bert_classifier_on_demo.predict_dist_batch(None, x)
        assert dist.shape[0] == 2 and dist.shape[1] == 2
        assert np.all(np.argmax(dist, axis=1) == y)

        pred = bert_classifier_on_demo.predict_batch(None, x)
        assert all([a == b for a, b in zip(pred, y)])

        pred = bert_classifier_on_demo.measure_batch(None, x)
        assert all([a == b for a, b in zip(pred, y)])
