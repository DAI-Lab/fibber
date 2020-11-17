from fibber.resources.resource_utils import (
    get_corenlp, get_glove_emb, get_nltk_data, get_stopwords, get_transformers,
    get_universal_sentence_encoder)


def download_all():
    get_nltk_data()
    get_glove_emb(download_only=True)
    get_transformers("bert-base-cased")
    get_transformers("bert-base-uncased")
    get_transformers("gpt2-medium")
    get_universal_sentence_encoder()
    get_corenlp()
    get_stopwords()


if __name__ == "__main__":
    download_all()
