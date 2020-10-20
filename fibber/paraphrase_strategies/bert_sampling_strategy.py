import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertTokenizer

from fibber import log
from fibber.paraphrase_strategies.strategy_base import StrategyBase
from fibber.paraphrase_strategies.wordpiece_emb import get_wordpiece_emb

logger = log.setup_custom_logger(__name__)

POST_PROCESSING_PATTERN = [
    (r"\s?'\s?t\s", "'t "),
    (r"\s?'\s?s\s", "'s "),
    (r"\s?'\s?ve\s", "'ve "),
    (r"\s?'\s?ll\s", "'ll "),
]

def post_process_text(text):
    """Postprocessing the text using regex patterns.

    Args:
        text (str): the str to be post processed.
    """
    for pattern in POST_PROCESSING_PATTERN:
        text = re.sub(pattern[0], pattern[1], text)
    return text


def tostring(tokenizer, seq):
    """Convert a sequence of word ids to a sentence. The post prossing is applied.

    Args:
        tokenizer (transformers.BertTokenizer): a BERT tokenizer.
        seq (list): a list-like sequence of word ids.
    """
    return post_process_text(
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(seq)))


def sample_word_from_logits(logits, temperature=1., top_k=0):
    """Sample a word from a distribution.

    Args:
        logits (torch.Tensor): tensor of logits with size ``(batch_size, vocab_size)``.
        temperature (float): the temperature of softmax. The PMF is
            ``softmax(logits/temperature)``.
        top_k (int): if ``k>0``, only sample from the top k most probable words.
    """
    logits = logits / temperature

    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    else:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    return idx


def all_accept_criteria(candidate_idxs, **kargs):
    """Always accept proposed words."""
    return candidate_idxs.detach().cpu().numpy()


def estimate_p(tokenizer, origin, batch_tensor, pos, idxs, use_metric,
               use_threshold, use_smoothing):
    """Estimate the unnormalized log pobability of a sentence, to be valid paraphrase.

    Args:
        tokenizer (transformers.BertTokenizer): a bert tokenizer.
        origin (str): original text.
        batch_tensor (np.array): tensor of a batch of text with size ``(batch_size, L)``.
        pos (int): the position to replace. Note that ``batch_tensor[:, pos]`` is not used.
        idxs (np.array): word ids with size ``(batch_size,)``.
        use_metric (USESemanticSimilarity): a universal sentence encoder metric object.
        use_threshold (float): the universal sentence encoder similarity threshold.
        use_smoothing (float): the smoothing parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    batch_tensor[:, pos] = idxs
    sentences = [tostring(tokenizer, x) for x in batch_tensor]
    use_semantic_similarity = use_metric.measure_batch(origin, sentences)
    return -use_smoothing * (
        np.maximum(use_threshold - np.asarray(use_semantic_similarity), 0) ** 2)


def similarity_accept_criteria(tokenizer, origin, batch_tensor, pos, previous_idxs, candidate_idxs,
                               use_metric, use_threshold, use_smoothing, weight, **kargs):
    """Accept or reject candidate word using the similarity as criteria.

    Args:
        tokenizer (transformers.BertTokenizer): a bert tokenizer.
        origin (str): original text.
        batch_tensor (torch.Tensor): tensor of a batch of text with size ``(batch_size, L)``.
        pos (int): the position to replace. Note that ``batch_tensor[:, pos]`` is not used.
        previous_idxs (torch.Tensor): word ids before current step of sampling with
            size ``(batch_size,)``.
        candidate_idxs (torch.Tensor): proposed word ids in this sampling step with
            size ``(batch_size,)``.
        use_metric (USESemanticSimilarity): a universal sentence encoder metric object.
        use_threshold (float): the universal sentence encoder similarity threshold.
        use_smoothing (float): the smoothing parameter for the criteria.

    Returns:
        (np.array): a 1-D int array of size `batch_size`. Each entry ``i`` is either
            ``previous_idxs[i]`` if rejected, or ``candidate_idxs[i]`` if accepted.
    """
    batch_tensor = batch_tensor.detach().cpu().numpy()
    previous_idxs = previous_idxs.detach().cpu().numpy()
    candidate_idxs = candidate_idxs.detach().cpu().numpy()

    unnormalized_log_p = estimate_p(
        tokenizer, origin, batch_tensor, pos, candidate_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    alpha = np.exp(unnormalized_log_p)
    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")
    idxs = candidate_idxs * accept + previous_idxs * (1 - accept)
    return idxs



def scmh_accept_criteria(tokenizer, origin, batch_tensor, pos, previous_idxs,
                         candidate_idxs, logits, temperature, use_metric,
                         use_threshold, use_smoothing, weight, **kargs):
    """Accept or reject the candidate word using the Metropolis hastings criteria.

    Args:
        tokenizer (transformers.BertTokenizer): a bert tokenizer.
        origin (str): original text.
        batch_tensor (torch.Tensor): tensor of a batch of text with size ``(batch_size, L)``.
        pos (int): the position to replace. Note that ``batch_tensor[:, pos]`` is not used.
        previous_idxs (torch.Tensor): word ids before current step of sampling with
            size ``(batch_size,)``.
        candidate_idxs (torch.Tensor): proposed word ids in this sampling step with
            size ``(batch_size,)``.
        logits (torch.Tensor): a 2-D float tensor of size `batch_size*vocab_size`.
        temperature (float): the temperature of the softmax.
        use_metric (USESemanticSimilarity): a universal sentence encoder metric object.
        use_threshold (float): the universal sentence encoder similarity threshold.
        use_smoothing (float): the smoothing parameter for the criteria.

    Returns:
        (np.array): a 1-D int array of size `batch_size`. Each entry ``i`` is either
            ``previous_idxs[i]`` if rejected, or ``candidate_idxs[i]`` if accepted.
    """
    batch_tensor = batch_tensor.detach().cpu().numpy()
    previous_idxs = previous_idxs.detach().cpu().numpy()
    candidate_idxs = candidate_idxs.detach().cpu().numpy()
    logits = (logits / temperature).detach().cpu().numpy()

    unnormalized_log_p_previous = estimate_p(
        tokenizer, origin, batch_tensor, pos, previous_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    unnormalized_log_p_candidate = estimate_p(
        tokenizer, origin, batch_tensor, pos, candidate_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    alpha = np.exp(unnormalized_log_p_candidate - unnormalized_log_p_previous
                    + logits[np.arange(len(previous_idxs)), previous_idxs]
                    - logits[np.arange(len(candidate_idxs)), candidate_idxs])

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")
    idxs = candidate_idxs * accept + previous_idxs * (1 - accept)
    return idxs

def none_constraint(**kargs):
    return 0

def allow_list_constraint(allow_list, **kargs):
    return -1e6 * (1 - allow_list)

def wpe_constraint(target_emb, word_embs, batch_tensor, pos,
                    wpe_threshold, wpe_smoothing, **kargs):
    current_emb = (word_embs(batch_tensor[:, 1:-1]).sum(dim=1)
                   - word_embs(batch_tensor[:, pos]))
    candidate_emb = current_emb[:, None, :] + word_embs.weight.data[None, :, :]
    dis = F.cosine_similarity(candidate_emb, target_emb[:, None, :], dim=2)
    dis = (wpe_threshold - dis).clamp_(min=0)
    return -wpe_smoothing * dis * dis

class BertSamplingStrategy(StrategyBase):
    __abbr__ = "bss"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("burnin_steps", int, 250, "number of burnin steps."),
        ("sampling_steps", int, 500, "number of sampling steps (including) burnin."),
        ("top_k", int, 100, "sample from top k words after burnin. Use 0 for all words."),
        ("temperature", float, 1., "the softmax temperature for sampling."),
        ("use_threshold", float, 0.9, "the threshold for USE similarity."),
        ("use_smoothing", float, 100, "the smoothing parameter for USE similarity."),
        ("wpe_threshold", float, 0.9, "the threshold for USE similarity."),
        ("wpe_smoothing", float, 1000, "the smoothing parameter for USE similarity."),
        ("burnin_add_schedule", str, "linear", ("the schedule decides how much additional "
            "constraint is added. options are linear, 0, 1.")),
        ("accept_criteria", str, "all", ("select an accept criteria for candidate words from "
            "all, similarity, scmh.")),
        ("additional_constraint", str, "none", ("select an additional constraint for candidate "
            "words from none, allow_list, wpe")),
        ("burnin_criteria_schedule", str, "1", ("the schedule decides how strict the criteria is "
            "used. options are linear, 0, 1."))
    ]

    def __repr__(self):
        return "%s-constraint_%s-criteria_%s" % (
            self.__class__.__name__,
            self._strategy_config["additional_constraint"],
            self._strategy_config["accept_criteria"])

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for BertSamplingStrategy.")
        if trainset["cased"]:
            model_init = "bert-base-cased"
        else:
            model_init = "bert-base-uncased"
        self._bert_lm = BertForMaskedLM.from_pretrained(model_init).to(self._device)
        self._tokenizer = BertTokenizer.from_pretrained(model_init)
        self._bert_lm.eval()
        for item in self._bert_lm.parameters():
            item.requires_grad = False

        # find the use metric
        self._use_metric = self._metric_bundle.get_metric("USESemanticSimilarity")

        # load word piece embeddings.
        wpe = get_wordpiece_emb(self._output_dir, self._dataset_name, trainset, self._device)
        self._word_embs = nn.Embedding(self._tokenizer.vocab_size, 300)
        self._word_embs.weight.data = torch.tensor(wpe.T).float()
        self._word_embs = self._word_embs.to(self._device)
        self._word_embs.eval()
        for item in self._word_embs.parameters():
            item.requires_grad = False

        # config _accept_fn and _additional_constraint_fn
        if self._strategy_config["accept_criteria"] == "all":
            self._accept_fn = all_accept_criteria
        elif self._strategy_config["accept_criteria"] == "similarity":
            self._accept_fn = similarity_accept_criteria
        elif self._strategy_config["accept_criteria"] == "scmh":
            self._accept_fn = scmh_accept_criteria
        else:
            assert 0

        if self._strategy_config["additional_constraint"] == "none":
            self._additional_constraint_fn = none_constraint
        elif self._strategy_config["additional_constraint"] == "wpe":
            self._additional_constraint_fn = wpe_constraint
        elif self._strategy_config["additional_constraint"] == "allow_list":
            self._additional_constraint_fn = allow_list_constraint
        else:
            assert 0


    def _parallel_sequential_generation(self, seed, batch_size):
        seq = ["[CLS]"] + self._tokenizer.tokenize(seed) + ["[SEP]"]

        batch_tensor = torch.tensor(
            [self._tokenizer.convert_tokens_to_ids(seq)] * batch_size).to(self._device)

        target_emb = self._word_embs(batch_tensor[:, 1:-1]).sum(dim=1)

        allow_list = F.one_hot(batch_tensor[0][1:-1], len(self._tokenizer.vocab)).sum(dim=0)

        for ii in range(self._strategy_config["sampling_steps"]):
            kk = np.random.randint(1, len(seq) - 1)
            previous_idxs = batch_tensor[:, kk].clone()
            batch_tensor[:, kk] = self._tokenizer.mask_token_id

            logits_lm = self._bert_lm(batch_tensor)[0][:, kk]
            logits_add = self._additional_constraint_fn(
                # for wpe constraint
                target_emb=target_emb,
                word_embs=self._word_embs,
                batch_tensor=batch_tensor,
                pos=kk,
                wpe_threshold=self._strategy_config["wpe_threshold"],
                wpe_smoothing=self._strategy_config["wpe_smoothing"],
                # for naive constraint
                allow_list=allow_list
            )

            if ii < self._strategy_config["burnin_steps"]:
                if self._strategy_config["burnin_add_schedule"] == "0":
                    logits_joint = logits_lm
                elif self._strategy_config["burnin_add_schedule"] == "1":
                    logits_joint = logits_lm + logits_add
                elif self._strategy_config["burnin_add_schedule"] == "linear":
                    logits_joint = logits_lm + (
                        ii / self._strategy_config["burnin_steps"]) * logits_add
                else:
                    assert 0
            else:
                logits_joint = logits_lm + logits_add

            top_k = self._strategy_config["top_k"] if (
                ii >= self._strategy_config["burnin_steps"]) else 0

            candidate_idxs = sample_word_from_logits(
                logits_joint, top_k=top_k, temperature=self._strategy_config["temperature"])


            if ii < self._strategy_config["burnin_steps"]:
                if self._strategy_config["burnin_criteria_schedule"] == "0":
                    accept_weight = 0
                elif self._strategy_config["burnin_criteria_schedule"] == "1":
                    accept_weight = 1
                elif self._strategy_config["burnin_criteria_schedule"] == "linear":
                    accept_weight = ii / self._strategy_config["burnin_steps"]
                else:
                    assert 0
            else:
                accept_weight = 1

            final_idxs = self._accept_fn(
                tokenizer=self._tokenizer,
                origin=seed,
                batch_tensor=batch_tensor,
                pos=kk,
                previous_idxs=previous_idxs,
                candidate_idxs=candidate_idxs,
                logits=logits_joint,
                use_metric=self._use_metric,
                use_threshold=self._strategy_config["use_threshold"],
                use_smoothing=self._strategy_config["use_smoothing"],
                weight=accept_weight)

            batch_tensor[:, kk] = torch.tensor(final_idxs).to(self._device)

        batch_tensor = batch_tensor.detach().cpu().numpy()
        return [tostring(self._tokenizer, x[1:-1]) for x in batch_tensor]

    def paraphrase_example(self, data_record, field_name, n):
        assert field_name == "text0"

        batch_size = self._strategy_config["batch_size"]

        sentences = []
        n_batches = math.ceil(n / batch_size)
        last_batch_size = n % batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size

        for id in range(n_batches):
            batch = self._parallel_sequential_generation(
                data_record[field_name], batch_size if id != n_batches - 1 else last_batch_size)
            sentences += batch

        assert len(sentences) == n
        return sentences[:n]
