import pytest

from fibber.paraphrase_strategies.bert_sampling_utils_text_parser import TextParser


@pytest.fixture()
def text_parser():
    text_parser = TextParser()
    return text_parser


def test_text_parser_split_to_sentence(text_parser):
    doc = ""
    result = text_parser.split_paragraph_to_sentences(doc)

    assert isinstance(result, list) and len(result) == 0

    doc = "hello world! Hello!"
    reference = ["hello world !", "Hello !"]
    result = text_parser.split_paragraph_to_sentences(doc)
    assert all([x == y for x, y in zip(result, reference)])
