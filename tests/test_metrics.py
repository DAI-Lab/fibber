from fibber.metrics.editing_distance import EditingDistance
from fibber.metrics.glove_semantic_similarity import GloVeSemanticSimilarity
from fibber.metrics.gpt2_grammar_quality import GPT2GrammarQuality
from fibber.metrics.metric_utils import MetricBundle
from fibber.metrics.use_semantic_similarity import USESemanticSimilarity


def test_editing_distance():
    editing_distance_measure = EditingDistance()

    s1 = "aa bb xx dd zz xx ee"
    s2 = "aa bb xy yz dd xy ee"
    assert editing_distance_measure(s1, s2) == 4

    s1 = "aa bb cc dd"
    s2 = "aa bb dd"
    assert editing_distance_measure(s1, s2) == 1

    s1 = "aa bb cc"
    s2 = "aa bb cc"
    assert editing_distance_measure(s1, s2) == 0

    s1 = "a. b; c?"
    s2 = "a1, b& c'"
    assert editing_distance_measure(s1, s2) == 1


def test_use_semantic_similarity():
    use_semantic_similarity_measure = USESemanticSimilarity()
    s1 = "Sunday is the first day in a week."
    s2 = "Obama was the president of the United State."
    assert use_semantic_similarity_measure(s1, s2) < 0.5

    s1 = "Saturday is the last day in a week"
    s2 = "Sunday is the last day in a week"
    assert use_semantic_similarity_measure(s1, s2) > 0.6


def test_gpt2_grammar_quality():
    gpt2_grammar_quality_measure = GPT2GrammarQuality()
    s1 = "Saturday is the last day in a week."
    s2 = "Sunday is is is first daay in a week."
    assert gpt2_grammar_quality_measure(s1, s2) > 5

    s1 = "Sunday is the last day in a week."
    s2 = "Saturday is the last day in a week."
    assert gpt2_grammar_quality_measure(s1, s2) < 2


def test_glove_semantic_similarity():
    glove_semantic_similarity_measure = GloVeSemanticSimilarity()

    s1 = "the a the to"
    s2 = "he him"
    assert glove_semantic_similarity_measure(s1, s2) == 0

    s1 = "Saturday is the last day in a week"
    s2 = "Sunday is the last day in a week"
    assert 0.95 < glove_semantic_similarity_measure(s1, s2) < 1


def test_metric_bundle():
    metric_bundle = MetricBundle()
    s1 = "Saturday is the last day in a week"
    s2 = "Sunday is the last day in a week"
    results = metric_bundle(s1, s2)
    assert len(results) == 4
