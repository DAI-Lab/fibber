import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertTokenizer

from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase, post_process_text
from fibber.paraphrase_strategies.wordpiece_emb import get_wordpiece_emb

logger = log.setup_custom_logger(__name__)


P_SMOOTHING = 100

def tostring(tokenizer, seq):
    return post_process_text(
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(seq)))


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



def accept_or_reject(tokenizer, origin, batch_tensor, pos, previous_idxs,
                     candidate_idxs, logpdf, use_metric):
    """Accept or reject the candidate word using the Metropolis hastings critera.

    Args:
        tokenizer (object): a bert tokenizer.
        origin (str): original text.
        batch_tensor (object): a 2-D int tensor of size `batch_size * L`.
        pos (int): the position to replace. The value of `batch_tensor[:, pos]` is not used.
        previous_idxs (object): a 1-D int tensor of size `batch_size`.
        candidate_idxs (object): a 1-D int tensor of size `batch_size`.
        logpdf (object): a 2-D float tensor of size `batch_size*vocab_size`.
        use_metric (object): a universal sentence encoder metric object.

    Returns:
        (object): a 1-D int array of size `batch_size`.
    """
    batch_tensor = batch_tensor.detach().cpu().numpy()
    previous_idxs = previous_idxs.detach().cpu().numpy()
    candidate_idxs = candidate_idxs.detach().cpu().numpy()
    logpdf = logpdf.detach().cpu().numpy()

    batch_tensor[:, pos] = previous_idxs
    sentences = [tostring(tokenizer, x) for x in batch_tensor]
    use_semantic_similarity = use_metric.batch_call(origin, sentences)
    unnormalized_log_p_previous = -P_SMOOTHING * np.maximum(
        0.9 - np.asarray(use_semantic_similarity), 0)

    batch_tensor[:, pos] = candidate_idxs
    sentences = [tostring(tokenizer, x) for x in batch_tensor]
    use_semantic_similarity = use_metric.batch_call(origin, sentences)
    unnormalized_log_p_candidate = -P_SMOOTHING * np.maximum(
        0.9 - np.asarray(use_semantic_similarity), 0)

    # print(unnormalized_log_p_candidate - unnormalized_log_p_previous)
    # print(logpdf[np.arange(len(previous_idxs)), previous_idxs])
    # print(logpdf[np.arange(len(candidate_idxs)), candidate_idxs])

    log_alpha = np.exp(unnormalized_log_p_candidate - unnormalized_log_p_previous
                        + logpdf[np.arange(len(previous_idxs)), previous_idxs]
                        - logpdf[np.arange(len(candidate_idxs)), candidate_idxs])

    # print(log_alpha)
    accept = np.asarray(np.random.rand(len(log_alpha)) < log_alpha, dtype="int32")
    # print(accept)
    idxs = candidate_idxs * accept + previous_idxs * (1 - accept)
    # print(idxs)
    # assert 0
    return idxs

class BertSamplingStrategy(StrategyBase):

    def __init__(self, FLAGS, metric_bundle):
        """Initialize the strategy."""
        super(SCMHSamplingWPECStrategy, self).__init__(FLAGS, metric_bundle)

        self._batch_size = 20
        self._top_k = 100
        self._burnin = 250
        self._max_iter = 500
        self._temperature = 1.0

        self._wpe_eps = 0.95
        self._wpe_smooth = 1000

        self._output_dir = FLAGS.output_dir
        self._dataset_name = FLAGS.dataset

        self._use_metric = metric_bundle.get_metric("USESemanticSimilarity")

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
            previous_idxs = batch_tensor[:, kk].clone()
            batch_tensor[:, kk] = self._tokenizer.mask_token_id

            out = self._bert_lm(batch_tensor)[0]
            current_emb = (self._word_embs(batch_tensor[:, 1:-1]).sum(dim=1)
                           - self._word_embs(batch_tensor[:, kk]))
            candidate_emb = current_emb[:, None, :] + self._word_embs.weight.data[None, :, :]
            dis = F.cosine_similarity(candidate_emb, target_emb[:, None, :], dim=2)
            dis = (self._wpe_eps - dis).clamp_(min=0)
            logpdf = -self._wpe_smooth * dis

            if ii >= self._burnin:
                logpdf_joint = out[:, kk] + logpdf
            else:
                logpdf_joint = out[:, kk] + ii / self._burnin * logpdf

            top_k = self._top_k if (ii >= self._burnin) else 0

            candidate_idxs = generate_step(logpdf_joint, top_k=top_k,
                                            temperature=self._temperature)

            idxs_prime = accept_or_reject(
                self._tokenizer, seed, batch_tensor, kk, previous_idxs, candidate_idxs,
                logpdf_joint, self._use_metric)
            batch_tensor[:, kk] = torch.tensor(idxs_prime).to(self._device)

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
