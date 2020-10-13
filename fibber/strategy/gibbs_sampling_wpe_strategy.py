import textattack
import torch
import math
import numpy as np

from .. import log
from .strategy_base import StrategyBase
from transformers import BertForMaskedLM, BertTokenizer
from .wordpiece_emb import get_wordpiece_emb
from torch import nn
from torch.nn import functional as F

logger = log.setup_custom_logger(__name__)

def tostring(tokenizer, seq):
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(seq))


def generate_step(logits, temperature=None, top_k=0):
    """ Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
    """

    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    else:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    return idx

class GibbsSamplingWPEStrategy(StrategyBase):

    def __init__(self, FLAGS, measurement_bundle):
        """Initialize the strategy."""
        super(GibbsSamplingWPEStrategy, self).__init__(FLAGS, measurement_bundle)

        self._batch_size = 20
        self._top_k = 100
        self._burnin = 250
        self._max_iter = 500
        self._temperature = 1.0

        self._output_dir = FLAGS.output_dir
        self._dataset_name = FLAGS.dataset

    def fit(self, trainset):
        if trainset["cased"]:
            model_init = "bert-base-cased"
        else:
            model_init = "bert-base-uncased"

        logger.info("Load bert language model for GibbsSamplingWPEStrategy.")

        self._bert_lm = BertForMaskedLM.from_pretrained(model_init).to(self._device)
        self._tokenizer = BertTokenizer.from_pretrained(model_init)
        self._bert_lm.eval()
        for item in self._bert_lm.parameters():
            item.requires_grad = False

        wpe = get_wordpiece_emb(self._output_dir, self._dataset_name, trainset, self._device)
        self._word_embs = nn.Embedding(self._tokenizer.vocab_size, 300)
        self._word_embs.weight.data = torch.tensor(wpe.T).float()
        self._word_embs = self._word_embs.to(self._device)
        self._word_embs.eval()
        for item in self._word_embs.parameters():
            item.requires_grad = False

    def _parallel_sequential_generation(self, seed):
        seq = ["[CLS]"] + self._tokenizer.tokenize(seed) + ["[SEP]"]

        batch_tensor = torch.tensor(
            [self._tokenizer.convert_tokens_to_ids(seq)] * self._batch_size).to(self._device)

        target_emb = self._word_embs(batch_tensor[:, 1:-1]).sum(dim=1)

        for ii in range(self._max_iter):
            kk = np.random.randint(1, len(seq) - 1)
            batch_tensor[:, kk] = self._tokenizer.mask_token_id

            out = self._bert_lm(batch_tensor)[0]
            current_emb = (self._word_embs(batch_tensor[:, 1:-1]).sum(dim=1)
                            - self._word_embs(batch_tensor[:, kk]))
            candidate_emb = current_emb[:, None, :] + self._word_embs.weight.data[None, :, :]
            dis = F.cosine_similarity(candidate_emb, target_emb[:, None, :], dim=2)
            dis = (0.95 - dis).clamp_(min=0)
            logpdf = -100 * dis
            logpdf_joint = out[:, kk] + logpdf

            top_k = self._top_k if (ii >= self._burnin) else 0

            idxs = generate_step(logpdf_joint, top_k=top_k, temperature=self._temperature)

            batch_tensor[:, kk] = idxs

        batch_tensor = batch_tensor.detach().cpu().numpy()

        return [tostring(self._tokenizer, x[1:-1]) for x in batch_tensor]


    def paraphrase_example(self, data_record, field_name, n):
        assert field_name == "text0"

        sentences = []
        n_batches = math.ceil(n / self._batch_size)

        for batch in range(n_batches):
            batch = self._parallel_sequential_generation(
                data_record[field_name])
            sentences += batch

        return sentences[:n]
