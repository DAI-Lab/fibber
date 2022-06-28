"""This metric computes the perplexity ratio ppl(paraphrase) / ppl(original text).

The perplexity is estimated using GPT2 model. This metric can reveal the meaningfulness of a
sentence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from fibber import log, resources
from fibber.metrics.metric_base import MetricBase

logger = log.setup_custom_logger(__name__)


def make_input_output_pair(tokenizer, x):
    """Tokenize the text, then construct input and output for GPT2."""
    toks = tokenizer.encode(x, add_special_tokens=True)
    return [tokenizer.bos_token_id] + toks[:-1], toks


def make_batch(toks_list):
    """Convert multiple text to a batch tensor."""
    n = len(toks_list)
    max_len = max([len(x) for x in toks_list])

    ids = np.zeros((n, max_len), dtype='int')
    mask = np.zeros((n, max_len), dtype='int')

    for i, item in enumerate(toks_list):
        ids[i, :len(item)] = np.asarray(item)
        mask[i, :len(item)] = 1

    return ids, mask


class GPT2PerplexityMetric(MetricBase):
    """This metric computes the perplexity of paraphrased text divided by the perplexity of
    original text. The perplexity is measured using GPT2 model.
    """

    def __init__(self, gpt2_pretrained_model="gpt2-medium", gpt2_gpu_id=-1, **kwargs):
        """Initialize GPT2 model."""
        super(GPT2PerplexityMetric, self).__init__(**kwargs)

        logger.info("load gpt2 model.")
        self._tokenizer = GPT2TokenizerFast.from_pretrained(
            resources.get_transformers(gpt2_pretrained_model))
        if gpt2_gpu_id == -1:
            logger.warning("GPT2 metric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("GPT2 metric is running on GPU %d.", gpt2_gpu_id)
            self._device = torch.device("cuda:%d" % gpt2_gpu_id)

        self._model = GPT2LMHeadModel.from_pretrained(
            resources.get_transformers(gpt2_pretrained_model)).to(
            self._device)

    def _get_ppl(self, sentences):
        """Compute the perplexity of sentences."""
        input_output = [make_input_output_pair(self._tokenizer, x) for x in sentences]

        input, output = zip(*input_output)

        toks_input, mask = make_batch(input)
        toks_output, _ = make_batch(output)

        mask = torch.tensor(mask).to(self._device)
        toks_input = torch.tensor(toks_input).to(self._device)
        toks_output = torch.tensor(toks_output).to(self._device)

        with torch.no_grad():
            logits = self._model(toks_input, attention_mask=mask)[0]

        logpw = torch.gather(F.log_softmax(logits, dim=-1), dim=-1,
                             index=toks_output.unsqueeze(dim=2)).squeeze(dim=2)
        ppl = torch.exp(-(logpw * mask).sum(dim=1) / mask.sum(dim=1))
        ppl = ppl.detach().cpu().numpy()
        return ppl

    def _measure_batch(self, origin, paraphrase_list, data_record=None, use_ratio=False, **kwargs):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            use_ratio (bool): returns the perplexity ratio.

        Returns:
            (list): a list containing the USE similarity metric for each paraphrase.
        """
        if use_ratio:
            ppls = self._get_ppl([origin] + paraphrase_list)
            res = ppls[1:] / ppls[0]
        else:
            res = self._get_ppl(paraphrase_list)
        return [float(x) for x in res]

    def _measure_multiple_examples(self, origin_list, paraphrase_list,
                                   data_record_list=None, use_ratio=False, **kwargs):
        assert len(origin_list) == len(paraphrase_list)
        if use_ratio:
            ppls = self._get_ppl(origin_list + paraphrase_list)
            res = ppls[len(origin_list):] / ppls[:len(origin_list)]
        else:
            res = self._get_ppl(paraphrase_list)
        print(res)
        return [float(x) for x in res]

    def _measure_example(self, origin, paraphrase, data_record=None, use_ratio=False, **kwargs):
        """Compute the perplexity ratio.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            use_ratio (bool): returns the perplexity ratio.

        """
        if use_ratio:
            ppl = self._get_ppl([origin, paraphrase])
            res = float(ppl[1] / ppl[0])
        else:
            res = float(self._get_ppl([paraphrase])[0])

        return res
