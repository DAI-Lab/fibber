import random
import re
import socket

from stanza.server import CoreNLPClient

from fibber import log, resources

logger = log.setup_custom_logger(__name__)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def generate_phrases_by_dfs(node, level, phrase_level, result):
    """DFS a parsing tree and generate all phrases in the tree."""
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
    """Add a space after punctuation."""
    return re.sub(r"([.,?:!])([^\s])", r"\1 \2", x)


class TextParser(object):
    """TextParser uses stanford core nlp tool to parse text.

    Args:
        port(int): the port to launch a stanford core nlp client
    """

    def __init__(self, port=9000):
        resources.get_corenlp()
        while is_port_in_use(port):
            port += 1
        self._core_nlp_client = CoreNLPClient(
            annotators=['parse'], timeout=600000, memory='16G', be_quiet=True,
            endpoint="http://localhost:%d" % port)

    def _get_parse_tree(self, sentence):
        """Generate a parsing tree from a sentence."""
        annotation = self._core_nlp_client.annotate(sentence)
        if len(annotation.sentence) != 1:
            logger.warning("_get_parse_tree should take one sentence. but %s is given." % sentence)
        return annotation.sentence[0].parseTree

    def split_paragraph_to_sentences(self, paragraph):
        """Split a paragraph into a list of sentences.

        Args:
            paragraph (str): a paragraph of English text.
        Returns:
            ([str]): a list of sentences.
        """
        paragraph = preprocess(paragraph)
        annotation = self._core_nlp_client.annotate(paragraph)
        return [" ".join([token.word for token in sentence.token])
                for sentence in annotation.sentence]

    def get_phrases(self, sentence, phrase_level=2):
        """Split a sentence into phrases.

        Args:
            sentence (str): a sentence in English.
            phrase_level (int): larger value gives shorter phrases.
        Returns:
            ([str]): a list of phrases.
        """
        root = self._get_parse_tree(sentence)

        phrases = []
        generate_phrases_by_dfs(root, 0, phrase_level, phrases)
        return phrases

    def phrase_level_shuffle(self, paragraph, n):
        """For each sentence, randomly shuffle phrases in that sentence.

        Args:
            paraphrase (str): a paragraph in English.
            n (int): number of randomly shuffled paragraph to generate.
        Returns:
            ([str]): a list of ``n`` shuffled paragraphs.
        """
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
