from fibber.resources.resource_utils import (
    get_bert_clf_demo, get_bert_lm_demo, get_corenlp, get_glove_emb, get_nltk_data, get_stopwords,
    get_transformers, get_universal_sentence_encoder)


def download_all():
    get_nltk_data()
    get_glove_emb(download_only=True)
    get_transformers("bert-base-cased")
    get_transformers("bert-base-uncased")
    get_transformers("gpt2-medium")
    get_universal_sentence_encoder()
    get_corenlp()
    get_stopwords()


def download_resources_for_demo(demo_path="."):
    get_nltk_data()
    get_glove_emb(download_only=True)
    get_transformers("gpt2-medium")
    get_transformers("bert-base-uncased")
    get_universal_sentence_encoder()
    get_stopwords()

    get_bert_clf_demo()
    get_bert_lm_demo(demo_path)


if __name__ == "__main__":
    download_all()
