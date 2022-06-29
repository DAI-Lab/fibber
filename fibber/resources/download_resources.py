from fibber.resources.resource_utils import (
    get_bert_clf_demo, get_bert_lm_demo, get_glove_emb, get_nltk_data, get_stopwords,
    get_transformers, get_universal_sentence_encoder, get_wordpiece_emb_demo)


def download_all():
    get_nltk_data()
    get_glove_emb(download_only=True)
    get_transformers("bert-base-cased")
    get_universal_sentence_encoder()
    get_stopwords()


def download_resources_for_demo():
    get_nltk_data()
    get_universal_sentence_encoder()
    get_stopwords()

    get_bert_clf_demo()
    get_bert_lm_demo()
    get_wordpiece_emb_demo()


if __name__ == "__main__":
    download_all()
