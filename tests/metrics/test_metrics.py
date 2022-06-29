import pytest
import torch

from fibber.metrics.distance.edit_distance_metric import EditDistanceMetric
from fibber.metrics.fluency.gpt2_perplexity_metric import GPT2PerplexityMetric
from fibber.metrics.similarity.ce_similarity_metric import CESimilarityMetric
from fibber.metrics.similarity.glove_similarity_metric import GloVeSimilarityMetric
from fibber.metrics.similarity.use_similarity_metric import USESimilarityMetric


@pytest.fixture
def gpu_id():
    if torch.cuda.device_count() > 0:
        return 0
    return -1


def metric_test_helper(metric, io_pairs, batched_io_pairs, eps=0, **kwargs):
    for (origin, paraphrase), true_output in io_pairs:
        value = metric.measure_example(origin, paraphrase, **kwargs)
        assert isinstance(value, int) or isinstance(value, float)
        assert abs(value - true_output) <= eps

    for (origin, paraphrase_list), true_output_list in batched_io_pairs:
        values = metric.measure_batch(origin, paraphrase_list, **kwargs)
        assert isinstance(values, list)
        assert all([isinstance(x, int) or isinstance(x, float) for x in values])
        assert all([abs(output - true_output) <= eps
                    for output, true_output in
                    zip(values, true_output_list)])


def test_edit_distance_metric():
    io_pairs = [
        (("aa bb xx dd zz xx ee", "aa bb xy yz dd xy ee"), 4),
        (("aa bb cc dd", "aa bb dd"), 1),
        (("aa bb cc", " aa  bb  cc "), 0),
        (("a. b; c?", "a1, b& c'"), 1)
    ]
    batched_io_pairs = [
        (("aa bb cc", ["aa bb cc dd", "aa bb cc", "aa ee"]), [1, 0, 2])
    ]
    editing_distance_metric = EditDistanceMetric(field="text0")

    metric_test_helper(editing_distance_metric, io_pairs, batched_io_pairs)


@pytest.mark.slow
def test_use_semantic_similarity(gpu_id):
    io_pairs = [
        (("Sunday is the first day in a week.",
          "Obama was the president of the United State."), 0.171),
        (("Sunday is the first day in a week",
          "Saturday is the last day in a week"), 0.759)
    ]
    batched_io_pairs = [
        (("Sunday is the first day in a week",
          ["Obama was the president of the United State.",
           "Saturday is the last day in a week"]),
         [0.171, 0.759]),
    ]
    use_semantic_similarity_metric = USESimilarityMetric(use_gpu_id=gpu_id, field="text0")
    metric_test_helper(use_semantic_similarity_metric, io_pairs, batched_io_pairs, eps=0.01)


@pytest.mark.slow
def test_ce_semantic_similarity(gpu_id):
    io_pairs = [
        (("Sunday is the first day in a week.",
          "Obama was the president of the United State."), 0.009),
        (("Sunday is the first day in a week",
          "Saturday is the last day in a week"), 0.291)
    ]
    batched_io_pairs = [
        (("Sunday is the first day in a week",
          ["Obama was the president of the United State.",
           "Saturday is the last day in a week"]),
         [0.009, 0.291]),
    ]
    ce_semantic_similarity_metric = CESimilarityMetric(ce_gpu_id=gpu_id, field="text0")
    metric_test_helper(ce_semantic_similarity_metric, io_pairs, batched_io_pairs, eps=0.01)


@pytest.mark.slow
def test_gpt2_grammar_quality(gpu_id):
    io_pairs = [
        (("Sunday is the first day in a week.",
          "Sunday is is is first daay in a week."), 14.64),
        (("Sunday is the first day in a week.",
          "Saturday is the last day in a week."), 1.10)
    ]
    batched_io_pairs = [
        (("Sunday is the first day in a week.",
          ["Sunday is is is first daay in a week.",
           "Saturday is the last day in a week."]),
         [14.64, 1.10]),
    ]
    gpt2_grammar_quality_metric = GPT2PerplexityMetric(gpt2_gpu_id=gpu_id, field="text0")
    metric_test_helper(gpt2_grammar_quality_metric, io_pairs, batched_io_pairs, eps=0.1,
                       use_ratio=True)


@pytest.mark.slow
def test_glove_semantic_similarity():
    io_pairs = [
        (("the a the to", "he him"), 0.0),
        (("Saturday is the last day in a week.",
          "Sunday is the last day in a week."), 0.997),
        (("Saturday is the last day in a week.",
          "Obama was the president of the United State."), 0.678),
    ]
    batched_io_pairs = [
        (("Saturday is the last day in a week.",
          ["Sunday is the last day in a week.",
           "Obama was the president of the United State."]),
         [0.997, 0.678])
    ]
    glove_semantic_similarity_metric = GloVeSimilarityMetric(field="text0")
    metric_test_helper(glove_semantic_similarity_metric, io_pairs, batched_io_pairs, eps=0.01)
