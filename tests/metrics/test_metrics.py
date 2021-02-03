import pytest
import torch

from fibber.metrics.edit_distance_metric import EditDistanceMetric
from fibber.metrics.glove_semantic_similarity_metric import GloVeSemanticSimilarityMetric
from fibber.metrics.gpt2_grammar_quality_metric import GPT2GrammarQualityMetric
from fibber.metrics.sbert_semantic_similarity_metric import SBERTSemanticSimilarityMetric
from fibber.metrics.use_semantic_similarity_metric import USESemanticSimilarityMetric


@pytest.fixture
def gpu_id():
    if torch.cuda.device_count() > 0:
        return 0
    return -1


def metric_test_helper(metric, io_pairs, batched_io_pairs, eps=0):
    for (origin, paraphrase), true_output in io_pairs:
        value = metric.measure_example(origin, paraphrase)
        assert isinstance(value, int) or isinstance(value, float)
        assert abs(value - true_output) <= eps

    for (origin, paraphrase_list), true_output_list in batched_io_pairs:
        values = metric.measure_batch(origin, paraphrase_list)
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
    editing_distance_metric = EditDistanceMetric()

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
    use_semantic_similarity_metric = USESemanticSimilarityMetric(use_gpu_id=gpu_id)
    metric_test_helper(use_semantic_similarity_metric, io_pairs, batched_io_pairs, eps=0.01)


@pytest.mark.slow
def test_sbert_semantic_similarity(gpu_id):
    io_pairs = [
        (("Sunday is the first day in a week.",
          "Obama was the president of the United State."), 0.326),
        (("Sunday is the first day in a week",
          "Saturday is the last day in a week"), 0.619)
    ]
    batched_io_pairs = [
        (("Sunday is the first day in a week",
          ["Obama was the president of the United State.",
           "Saturday is the last day in a week"]),
         [0.326, 0.619]),
    ]
    sbert_semantic_similarity_metric = SBERTSemanticSimilarityMetric(sbert_gpu_id=gpu_id)
    metric_test_helper(sbert_semantic_similarity_metric, io_pairs, batched_io_pairs, eps=0.01)


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
    gpt2_grammar_quality_metric = GPT2GrammarQualityMetric(gpt2_gpu_id=gpu_id)
    metric_test_helper(gpt2_grammar_quality_metric, io_pairs, batched_io_pairs, eps=0.1)


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
    glove_semantic_similarity_metric = GloVeSemanticSimilarityMetric()
    metric_test_helper(glove_semantic_similarity_metric, io_pairs, batched_io_pairs, eps=0.01)
