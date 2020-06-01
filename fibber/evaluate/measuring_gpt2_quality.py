import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .. import log

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = log.setup_custom_logger('measure-gpt2')


def make_input_output_pair(tokenizer, x):
  toks = tokenizer.encode(x, add_special_tokens=True)
  return [tokenizer.bos_token_id] + toks[:-1], toks


def make_batch(toks_list):
  n = len(toks_list)
  max_len = max([len(x) for x in toks_list])

  ids = np.zeros((n, max_len), dtype='int')
  mask = np.zeros((n, max_len), dtype='int')

  for i, item in enumerate(toks_list):
    ids[i, :len(item)] = np.asarray(item)
    mask[i, :len(item)] = 1

  return ids, mask


class GPT2Quality(object):

  def __init__(self, pretrained="gpt2"):
    super(GPT2Quality, self).__init__()

    logger.info("load gpt2 model.")
    self._tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
    self._model = GPT2LMHeadModel.from_pretrained(pretrained).to(DEVICE)

  def __call__(self, s1, s2):
    s1_input, s1_output = make_input_output_pair(self._tokenizer, s1)
    s2_input, s2_output = make_input_output_pair(self._tokenizer, s2)

    toks_input, mask = make_batch([s1_input, s2_input])
    toks_output, _ = make_batch([s1_output, s2_output])

    mask = torch.tensor(mask).to(DEVICE)
    toks_input = torch.tensor(toks_input).to(DEVICE)
    toks_output = torch.tensor(toks_output).to(DEVICE)

    logits = self._model(toks_input, attention_mask=mask)[0]

    logpw = torch.gather(F.log_softmax(logits, dim=-1), dim=-1,
                         index=toks_output.unsqueeze(dim=2)).squeeze(dim=2)
    ppl = torch.exp(-(logpw * mask).sum(dim=1) / mask.sum(dim=1))
    ppl = ppl.detach().cpu().numpy()
    return ppl[1] / ppl[0]


if __name__ == "__main__":
  measure = GPT2Quality()

  s1 = "a is a a a a"
  s2 = "rose is a rose"
  print(s1, s2, measure(s1, s2), sep="\n")

  s1 = "Saturday is the last day in a week"
  s2 = "Sunday is the last day in a week"
  print(s1, s2, measure(s1, s2), sep="\n")
