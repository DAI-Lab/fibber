import random
import re

from stanza.server import CoreNLPClient

# get all phrases from a sentence


def generate_phrases_by_dfs(node, level, phrase_level, result):
    if len(node.child) == 0:
        # if current node is leaf
        word = node.value.strip()
        if level <= phrase_level:
            result.append(word)
        return word

    phrase = ""
    for child in node.child:
        phrase += " " + generate_phrases_by_dfs(child, level + 1, phrase_level, result)
    phrase = " ".join(phrase.split())

    if level == phrase_level:
        result.append(phrase)
    return phrase


def preprocess(x):
    return re.sub(r"([.,?:!])([^\s])", r"\1 \2", x)


class TextParser(object):
    """TextParser uses stanford core nlp tool to parse text."""

    def __init__(self, port=9000):
        self._core_nlp_client = CoreNLPClient(
            annotators=['parse'], timeout=30000, memory='16G', be_quiet=True,
            endpoint="http://localhost:%d" % port)

    def _get_parse_tree(self, sentence):
        annotation = self._core_nlp_client.annotate(sentence)
        if len(annotation.sentence) != 1:
            logger.warning("_get_parse_tree should take one sentence. but %s is given." % sentence)
        return annotation.sentence[0].parseTree

    def split_paragraph_to_sentences(self, paragraph):
        paragraph = preprocess(paragraph)
        annotation = self._core_nlp_client.annotate(paragraph)
        return [" ".join([token.word for token in sentence.token])
                for sentence in annotation.sentence]

    def get_phrases(self, sentence, phrase_level=2):
        root = self._get_parse_tree(sentence)

        phrases = []
        generate_phrases_by_dfs(root, 0, phrase_level, phrases)
        return phrases

    def phrase_level_shuffle(self, paragraph, n):
        paragraph = preprocess(paragraph)
        sentences = self.split_paragraph_to_sentences(paragraph)
        bins = []
        for sentence in sentences:
            bins.append(self.get_phrases(sentence))

        ret = []
        for i in range(n):
            sent = ""
            for bin in bins:
                bin_tmp = bin[:]
                random.shuffle(bin_tmp)
                sent += " " + " ".join(bin_tmp)

            ret.append(sent.strip())
        return ret
