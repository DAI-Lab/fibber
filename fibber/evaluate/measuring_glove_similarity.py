import numpy as np
import tqdm

from .. import log

logger = log.setup_custom_logger('measure-glove')


def load_glove_model(glove_file, dim=300):
  glove_file_lines = open(glove_file, 'r').readlines()

  emb_table = np.zeros((len(glove_file_lines), dim), dtype='float32')
  id_to_tok = []
  tok_to_id = {}

  logger.info("load glove embeddings for glove similarity measurement.")
  id = 0
  for line in tqdm.tqdm(glove_file_lines):
    split_line = line.split()
    word = split_line[0]
    emb_table[id] = np.array([float(val) for val in split_line[1:]])
    id_to_tok.append(word)
    tok_to_id[word] = id
    id += 1

  return emb_table, id_to_tok, tok_to_id


def compute_emb(emb_table, id_to_tok, tok_to_id, x):
  toks = x.split()
  embs = []
  for item in toks:
    if item.lower() in tok_to_id:
      embs.append(emb_table[tok_to_id[item.lower()]])
  return np.sum(embs, axis=0)


def compute_emb_sim(emb_table, id_to_tok, tok_to_id, x, y):
  ex = compute_emb(emb_table, id_to_tok, tok_to_id, x)
  ey = compute_emb(emb_table, id_to_tok, tok_to_id, y)
  return ((ex * ey).sum()
          / (np.linalg.norm(ex) + 1e-8)
          / (np.linalg.norm(ey) + 1e-8))


class GloVeSimilarity(object):
  def __init__(self, embfile, dim, stopfile):
    super(GloVeSimilarity, self).__init__()
    logger.info("load glove embeddings.")
    self._emb_table, self._id_to_tok, self._tok_to_id = load_glove_model(
        embfile, dim)

    with open(stopfile) as f:
      stopwords = f.readlines()

    for word in stopwords:
      word = word.lower().strip()
      if word in self._tok_to_id:
        self._emb_table[self._tok_to_id[word], :] = 0

  def __call__(self, s1, s2):
    return float(compute_emb_sim(self._emb_table, self._id_to_tok,
                           self._tok_to_id, s1, s2))


if __name__ == "__main__":
  measure = GloVeSimilarity("data/glove.6B.300d.txt", 300, "data/stopwords.txt")

  s1 = "the a the to"
  s2 = "he him"
  print(s1, s2, measure(s1, s2), sep="\n")

  s1 = "Saturday is the last day in a week"
  s2 = "Sunday is the last day in a week"
  print(s1, s2, measure(s1, s2), sep="\n")
