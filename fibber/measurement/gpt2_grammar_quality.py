import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .. import log
from .measurement_base import MeasurementBase

logger = log.setup_custom_logger('gpt2_grammar_quality')


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


class GPT2GrammarQuality(MeasurementBase):

    def __init__(self, gpt2_pretrained_model="gpt2", gpt2_gpu_id=None, **kargs):
        super(GPT2GrammarQuality, self).__init__()

        logger.info("load gpt2 model.")
        self._tokenizer = GPT2Tokenizer.from_pretrained(gpt2_pretrained_model)
        if gpt2_gpu_id is None:
            logger.warning("GPT2 measurement is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("GPT2 measurement is running on GPU %d.", gpt2_gpu_id)
            self._device = torch.device("cuda:%d" % gpt2_gpu_id)

        self._model = GPT2LMHeadModel.from_pretrained(gpt2_pretrained_model).to(self._device)

    def _get_ppl(self, origin, paraphrase):
        origin_input, origin_output = make_input_output_pair(self._tokenizer, origin)
        paraphrase_input, paraphrase_output = make_input_output_pair(self._tokenizer, paraphrase)

        toks_input, mask = make_batch([origin_input, paraphrase_input])
        toks_output, _ = make_batch([origin_output, paraphrase_output])

        mask = torch.tensor(mask).to(self._device)
        toks_input = torch.tensor(toks_input).to(self._device)
        toks_output = torch.tensor(toks_output).to(self._device)

        logits = self._model(toks_input, attention_mask=mask)[0]

        logpw = torch.gather(F.log_softmax(logits, dim=-1), dim=-1,
                             index=toks_output.unsqueeze(dim=2)).squeeze(dim=2)
        ppl = torch.exp(-(logpw * mask).sum(dim=1) / mask.sum(dim=1))
        ppl = ppl.detach().cpu().numpy()
        return ppl

    def __call__(self, origin, paraphrase):
        ppl = self._get_ppl(origin, paraphrase)
        return float(ppl[1] / ppl[0])
