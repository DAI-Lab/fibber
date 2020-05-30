from measuring_editing_distance import EditingDistance
from measuring_glove_similarity import GloVeSimilarity
from measuring_gpt2_quality import GPT2Quality
from measuring_use_similarity import USESimilarity

MEASURE_GLOVE = "glove_sim"
MEASURE_USE = "use_sim"
MEASURE_GPT2 = "gpt2_qua"
MEASURE_EDIT = "edit_dis"


def make_measures(datafolder,
                  use_glove=True,
                  use_gpt2=True,
                  use_editing=False,
                  use_use=False):
  measures = {}
  if use_glove:
    measures[MEASURE_GLOVE] = GloVeSimilarity(
        datafolder + "/glove.6B.300d.txt", 300, datafolder + "/stopwords.txt")
  if use_gpt2:
    measures[MEASURE_GPT2] = GPT2Quality()

  if use_editing:
    measures[MEASURE_EDIT] = EditingDistance()

  if use_use:
    measures[MEASURE_USE] = USESimilarity()

  return measures


def _evaluate(ori, adv, measures):
  ret = {}
  for name, measure in measures.items():
    ret[name] = measure(ori, adv)
  return ret

def evaluate(ori, adv, measures):
  if isinstance(ori, str):
    assert isinstance(adv, str)
    return _evaluate(ori, adv, measures)

  assert len(ori) == len(adv)
  ret = []
  for u, v in zip(ori, adv):
    ret.append(_evaluate(u, v, measures))
  return ret


if __name__ == "__main__":
  measures = make_measures("data/", use_editing=True, use_use=True)
  print(evaluate("Happy holiday .", "fibber fibber .", measures))

  print(evaluate(["rose is a rose", "Saturday is the last day in a week"],
                 ["a is a a a a", "Sunday is the last day in a week"],
                 measures))
