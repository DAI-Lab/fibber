from fibber.resources.download_resources import download_all, download_resources_for_demo
from fibber.resources.resource_utils import (
    get_bert_clf_demo, get_corenlp, get_glove_emb, get_nltk_data, get_stopwords, get_transformers,
    get_universal_sentence_encoder)

__all__ = [
    "get_glove_emb",
    "get_stopwords",
    "get_nltk_data",
    "get_universal_sentence_encoder",
    "get_transformers",
    "get_corenlp",
    "download_all",
    "download_resources_for_demo",
    "get_bert_clf_demo"]
