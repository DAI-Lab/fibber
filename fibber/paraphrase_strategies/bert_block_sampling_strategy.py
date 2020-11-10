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
from fibber.paraphrase_strategies.text_parser import TextParser
from fibber.paraphrase_strategies.lm import get_lm
logger = log.setup_custom_logger(__name__)

POST_PROCESSING_PATTERN = [
    (r"\s+n\s", "n "),
    (r"\s*'\s*t\s", "'t "),
    (r"\s*'\s*s\s", "'s "),
    (r"\s*'\s*ve\s", "'ve "),
    (r"\s*'\s*ll\s", "'ll "),
    (r"\s*n't\s", "n't "),
    (r"- -", "--"),
    (r"\s*([\.,?!])", r"\1"),
    (r"\s+-\s+", "-"),
]

PRE_PROCESSING_PATTERN = [
    (r"can't\s", r" cannot "),
    (r"won't\s", r" will not "),
    (r"n't\s", r" not "),
    (r"'ll\s", r" will "),
    (r"'ve\s", r" have "),
]

AUTO_SENTENCE_LEN_THRESHOLD = 50  # 50 words

STATS_TOTAL = 0
STATS_ACCEPT = 0

def process_text(text, patterns):
    """Postprocessing the text using regex patterns.

    Args:
        text (str): the str to be post processed.
    """
    for pattern in patterns:
        text = re.sub(pattern[0], pattern[1], text)
    return text


def tostring(tokenizer, seq):
    """Convert a sequence of word ids to a sentence. The post prossing is applied.

    Args:
        tokenizer (transformers.BertTokenizer): a BERT tokenizer.
        seq (list): a list-like sequence of word ids.
    """
    return process_text(
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(seq)), POST_PROCESSING_PATTERN)


def sample_word_from_logits(logits, temperature=1., top_k=0):
    """Sample a word from a distribution.

    Args:
        logits (torch.Tensor): tensor of logits with size ``(batch_size, vocab_size)``.
        temperature (float): the temperature of softmax. The PMF is
            ``softmax(logits/temperature)``.
        top_k (int): if ``k>0``, only sample from the top k most probable words.
    """
    bs = logits.shape[0]
    window_size = logits.shape[1]
    vocab_size = logits.shape[2]

    logits = logits.view(bs * window_size, vocab_size)
    logits = logits / temperature

    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    else:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    return idx.view(bs, window_size)


def all_accept_criteria(candidate_idxs, **kargs):
    """Always accept proposed words."""
    return candidate_idxs.detach().cpu().numpy()


def estimate_p(tokenizer, origin, batch_tensor, pos, pos_ed, idxs, use_metric,
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
    batch_tensor[:, pos:pos_ed] = idxs
    sentences = [tostring(tokenizer, x) for x in batch_tensor]
    use_semantic_similarity = use_metric.measure_batch(origin, sentences)
    return -use_smoothing * (
        np.maximum(use_threshold - np.asarray(use_semantic_similarity), 0) ** 2)

def estimate_p2(tokenizer, origin, batch_tensor, pos, pos_ed, idxs, use_metric,
               use_threshold, use_smoothing, ppl_metric, ppl_smoothing):
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
    batch_tensor[:, pos:pos_ed] = idxs
    sentences = [tostring(tokenizer, x) for x in batch_tensor]
    use_semantic_similarity = use_metric.measure_batch(origin, sentences)
    ppl_ratio = ppl_metric.measure_batch(origin, sentences)
    return (-use_smoothing * (
        np.maximum(use_threshold - np.asarray(use_semantic_similarity), 0) ** 2)
        -ppl_smoothing * np.maximum(np.asarray(ppl_ratio - 1), 0) ** 2)


def relative_similarity_accept_criteria(tokenizer, origin, batch_tensor, pos, pos_ed, previous_idxs, candidate_idxs,
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
    if weight == 0:
        return candidate_idxs
    unnormalized_log_p_candidate = estimate_p(
        tokenizer, origin, batch_tensor, pos, pos_ed, candidate_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    unnormalized_log_p_previous = estimate_p(
        tokenizer, origin, batch_tensor, pos, pos_ed, previous_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    alpha = np.exp(np.asarray(unnormalized_log_p_candidate < use_threshold, dtype="float32")
            * np.asarray(unnormalized_log_p_candidate < unnormalized_log_p_previous, dtype="float32")
            * unnormalized_log_p_candidate)
    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")
    idxs = candidate_idxs * accept.reshape(-1, 1) + previous_idxs * (1 - accept.reshape(-1, 1))
    return idxs

def relative_similarity_and_clf_accept_criteria(
    tokenizer, origin, batch_tensor, pos, pos_ed, previous_idxs, candidate_idxs,
    use_metric, use_threshold, use_smoothing, weight, clf_metric, label, clf_weight,
    field_name, data_record, **kargs):
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
    if weight == 0:
        return candidate_idxs

    if field_name == "text1":
        context = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data_record["text0"]))
        context = np.asarray([context] * batch_tensor.shape[0], dtype="int64")
        l1 = context.shape[1]
        batch_tensor[:, 0] = tokenizer.sep_token_id
        batch_tensor = np.concatenate([context, batch_tensor], axis=1)
        tok_type = np.zeros_like(batch_tensor)
        tok_type[l1+1:] = 1
        pos += l1
        pos_ed += l1
    else:
        assert field_name == "text0"
        tok_type = np.zeros_like(batch_tensor)



    unnormalized_log_p_candidate = estimate_p(
        tokenizer, origin, batch_tensor, pos, pos_ed, candidate_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    batch_tensor[:, pos:pos_ed] = candidate_idxs
    clf_pred_candidate = F.log_softmax(
        clf_metric._model(torch.tensor(batch_tensor).to(clf_metric._device),
                            token_type_ids=torch.tensor(tok_type).to(clf_metric._device)
        )[0], dim=-1).detach().cpu().numpy()

    unnormalized_log_p_previous = estimate_p(
        tokenizer, origin, batch_tensor, pos, pos_ed, previous_idxs,
        use_metric, use_threshold, weight * use_smoothing)

    batch_tensor[:, pos:pos_ed] = previous_idxs
    clf_pred_previous = F.log_softmax(
        clf_metric._model(torch.tensor(batch_tensor).to(clf_metric._device),
            token_type_ids=torch.tensor(tok_type).to(clf_metric._device)
        )[0], dim=-1).detach().cpu().numpy()

    def score(x):
        correct_pred = (x[:, label]).copy()
        x[:, label] = -1e8
        next_max = np.max(x, axis=1)
        return np.minimum(next_max - correct_pred, 0)

    log_alpha_1 = (np.asarray(unnormalized_log_p_candidate < use_threshold, dtype="float32")
        * np.asarray(unnormalized_log_p_candidate < unnormalized_log_p_previous, dtype="float32")
        * unnormalized_log_p_candidate)
    log_alpha_2 = weight * clf_weight * (score(clf_pred_candidate) - score(clf_pred_previous))

    alpha = np.exp(log_alpha_1 + log_alpha_2)

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")
    idxs = candidate_idxs * accept.reshape(-1, 1) + previous_idxs * (1 - accept.reshape(-1, 1))
    return idxs


def relative_similarity_and_ppl_and_clf_accept_criteria(
    tokenizer, origin, batch_tensor, pos, pos_ed, previous_idxs, candidate_idxs,
    use_metric, use_threshold, use_smoothing, weight, clf_metric, label, clf_weight,
    field_name, data_record, ppl_metric, ppl_smoothing, **kargs):
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
    global STATS_TOTAL
    global STATS_ACCEPT
    batch_tensor = batch_tensor.detach().cpu().numpy()
    previous_idxs = previous_idxs.detach().cpu().numpy()
    candidate_idxs = candidate_idxs.detach().cpu().numpy()
    if weight == 0:
        return candidate_idxs

    if field_name == "text1":
        context = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data_record["text0"]))
        context = np.asarray([context] * batch_tensor.shape[0], dtype="int64")
        l1 = context.shape[1]
        batch_tensor[:, 0] = tokenizer.sep_token_id
        batch_tensor = np.concatenate([context, batch_tensor], axis=1)
        tok_type = np.zeros_like(batch_tensor)
        tok_type[l1+1:] = 1
        pos += l1
        pos_ed += l1
    else:
        assert field_name == "text0"
        tok_type = np.zeros_like(batch_tensor)



    unnormalized_log_p_candidate = estimate_p2(
        tokenizer, origin, batch_tensor, pos, pos_ed, candidate_idxs,
        use_metric, use_threshold, weight * use_smoothing, ppl_metric, weight * ppl_smoothing)

    batch_tensor[:, pos:pos_ed] = candidate_idxs
    clf_pred_candidate = F.log_softmax(
        clf_metric._model(torch.tensor(batch_tensor).to(clf_metric._device),
                            token_type_ids=torch.tensor(tok_type).to(clf_metric._device)
        )[0], dim=-1).detach().cpu().numpy()

    unnormalized_log_p_previous = estimate_p2(
        tokenizer, origin, batch_tensor, pos, pos_ed, previous_idxs,
        use_metric, use_threshold, weight * use_smoothing, ppl_metric, weight * ppl_smoothing)

    batch_tensor[:, pos:pos_ed] = previous_idxs
    clf_pred_previous = F.log_softmax(
        clf_metric._model(torch.tensor(batch_tensor).to(clf_metric._device),
            token_type_ids=torch.tensor(tok_type).to(clf_metric._device)
        )[0], dim=-1).detach().cpu().numpy()

    def score(x):
        correct_pred = (x[:, label]).copy()
        x[:, label] = -1e8
        next_max = np.max(x, axis=1)
        return np.minimum(next_max - correct_pred, 0)

    log_alpha_1 = (np.asarray(unnormalized_log_p_candidate < use_threshold, dtype="float32")
        * np.asarray(unnormalized_log_p_candidate < unnormalized_log_p_previous, dtype="float32")
        * unnormalized_log_p_candidate)
    log_alpha_2 = weight * clf_weight * (score(clf_pred_candidate) - score(clf_pred_previous))

    alpha = np.exp(log_alpha_1 + log_alpha_2)

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")

    STATS_TOTAL += len(accept)
    STATS_ACCEPT += np.sum(accept)
    idxs = candidate_idxs * accept.reshape(-1, 1) + previous_idxs * (1 - accept.reshape(-1, 1))
    return idxs


def none_constraint(**kargs):
    return 0

def allow_list_constraint(allow_list, **kargs):
    return -1e6 * (1 - allow_list)

def wpe_constraint(target_emb, word_embs, batch_tensor, pos, pos_ed,
                    wpe_threshold, wpe_smoothing, **kargs):
    current_emb = (word_embs(batch_tensor[:, 1:-1]).sum(dim=1)
                   - word_embs(batch_tensor[:, pos:pos_ed]).sum(dim=1))
    candidate_emb = current_emb[:, None, :] + word_embs.weight.data[None, :, :]
    dis = F.cosine_similarity(candidate_emb, target_emb[:, None, :], dim=2)
    dis = (wpe_threshold - dis).clamp_(min=0).unsqueeze(1)
    return -wpe_smoothing * dis * dis

class BertBlockSamplingStrategy(StrategyBase):
    __abbr__ = "bbss"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 1, "the block sampling window size."),
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
            "all, similarity, relative_similarity, scmh.")),
        ("additional_constraint", str, "none", ("select an additional constraint for candidate "
            "words from none, allow_list, wpe")),
        ("burnin_criteria_schedule", str, "1", ("the schedule decides how strict the criteria is "
            "used. options are linear, 0, 1.")),
        ("seed_option", str, "origin", ("the option for seed sentences in generation. "
            "choose from origin, auto.")),
        ("split_sentence", str, "0", "split paragraph to sentence."),
        ("sentence_len_threshold", int, 10, ("if split paragraph, sentence shorter than threshold "
            "is not paraphrased.")),
        ("stanza_port", int, 9000, "stanza port"),
        ("lm_option", str, "pretrain", "choose from pretrain, finetune, adv."),
        ("lm_steps", int, 5000, "lm training steps."),
        ("clf_weight", float, 10, "weight for the clf score in the criteria."),
        ("ppl_smoothing", float, 1, "the smoothing parameter for ppl."),
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
        self._tokenizer = BertTokenizer.from_pretrained(model_init)

        if self._strategy_config["lm_option"] == "pretrain":
            self._bert_lm = BertForMaskedLM.from_pretrained(model_init).to(self._device)
            self._bert_lm.eval()
            for item in self._bert_lm.parameters():
                item.requires_grad = False
        elif self._strategy_config["lm_option"] == "finetune":
            self._bert_lm = get_lm(
                self._output_dir, trainset, -1, self._device, self._strategy_config["lm_steps"])
            self._bert_lm.eval()
            self._bert_lm.to(self._device)
            for item in self._bert_lm.parameters():
                item.requires_grad = False
        elif self._strategy_config["lm_option"] == "adv":
            self._bert_lms = []
            for i in range(len(trainset["label_mapping"])):
                lm = get_lm(
                    self._output_dir, trainset, i, self._device, self._strategy_config["lm_steps"])
                lm.eval()
                for item in lm.parameters():
                    item.requires_grad = False
                self._bert_lms.append(lm)
        else:
            assert 0

        # find the use metric
        self._use_metric = self._metric_bundle.get_metric("USESemanticSimilarity")
        self._clf_metric = self._metric_bundle.get_metric("BertClfPrediction")
        self._ppl_metric = self._metric_bundle.get_metric("GPT2GrammarQuality")

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
        elif self._strategy_config["accept_criteria"] == "relative_similarity":
            self._accept_fn = relative_similarity_accept_criteria
        elif self._strategy_config["accept_criteria"] == "relative_similarity_and_clf":
            self._accept_fn = relative_similarity_and_clf_accept_criteria
        elif self._strategy_config["accept_criteria"] == "relative_similarity_and_ppl_and_clf_accept":
            self._accept_fn = relative_similarity_and_ppl_and_clf_accept_criteria
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

        # load text parser
        self._text_parser = TextParser(self._strategy_config["stanza_port"])


    def _parallel_sequential_generation(self, seed, batch_size, burnin_steps, sampling_steps,
                                        field_name, data_record):
        if self._strategy_config["seed_option"] == "origin":
            seq = ["[CLS]"] + self._tokenizer.tokenize(seed) + ["[SEP]"]
            batch_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(seq)] * batch_size).to(self._device)
            tensor_len = len(seq)
        elif self._strategy_config["seed_option"] == "auto":
            seeds = self._text_parser.phrase_level_shuffle(seed, batch_size)
            seq = [self._tokenizer.tokenize(x) for x in seeds]
            tensor_len = max([len(x) for x in seq])
            seq = [["[CLS]"] + x + ["[MASK]"] * (tensor_len - len(x)) + ["[SEP]"] for x in seq]
            tensor_len += 2
            batch_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(x) for x in seq]).to(self._device)
        else:
            assert 0

        target_emb = self._word_embs(batch_tensor[:, 1:-1]).sum(dim=1)

        allow_list = F.one_hot(batch_tensor[0][1:-1], len(self._tokenizer.vocab)).sum(dim=0)

        for ii in range(sampling_steps):
            kk = np.random.randint(1, tensor_len - 1)
            kk_ed = min(kk + self._strategy_config["window_size"], tensor_len - 1)

            previous_idxs = batch_tensor[:, kk:kk_ed].clone()
            batch_tensor[:, kk:kk_ed] = self._tokenizer.mask_token_id

            od = np.arange(kk, kk_ed)
            np.random.shuffle(od)
            for t in od:
                logits_lm = self._bert_lm(batch_tensor)[0][:, t:t+1]
                logits_add = self._additional_constraint_fn(
                    # for wpe constraint
                    target_emb=target_emb,
                    word_embs=self._word_embs,
                    batch_tensor=batch_tensor,
                    pos=t,
                    pos_ed=t+1,
                    wpe_threshold=self._strategy_config["wpe_threshold"],
                    wpe_smoothing=self._strategy_config["wpe_smoothing"],
                    # for naive constraint
                    allow_list=allow_list
                )

                if ii < burnin_steps:
                    if self._strategy_config["burnin_add_schedule"] == "0":
                        logits_joint = logits_lm
                    elif self._strategy_config["burnin_add_schedule"] == "1":
                        logits_joint = logits_lm + logits_add
                    elif self._strategy_config["burnin_add_schedule"] == "linear":
                        logits_joint = logits_lm + (
                            ii / burnin_steps) * logits_add
                    else:
                        assert 0
                else:
                    logits_joint = logits_lm + logits_add

                top_k = self._strategy_config["top_k"] if (
                    ii >= burnin_steps) else 100

                candidate_idxs = sample_word_from_logits(
                    logits_joint, top_k=top_k, temperature=self._strategy_config["temperature"])
                batch_tensor[:, t:t+1] = candidate_idxs

            candidate_idxs = batch_tensor[:, kk:kk_ed].clone()

            if ii < burnin_steps:
                if self._strategy_config["burnin_criteria_schedule"] == "0":
                    accept_weight = 0
                elif self._strategy_config["burnin_criteria_schedule"] == "1":
                    accept_weight = 1
                elif self._strategy_config["burnin_criteria_schedule"] == "linear":
                    accept_weight = ii / burnin_steps
                else:
                    assert 0
            else:
                accept_weight = 1

            final_idxs = self._accept_fn(
                tokenizer=self._tokenizer,
                origin=seed,
                batch_tensor=batch_tensor,
                pos=kk,
                pos_ed=kk_ed,
                previous_idxs=previous_idxs,
                candidate_idxs=candidate_idxs,
                logits=logits_joint,
                use_metric=self._use_metric,
                use_threshold=self._strategy_config["use_threshold"],
                use_smoothing=self._strategy_config["use_smoothing"],
                weight=accept_weight,
                ## with clf:
                clf_metric=self._clf_metric,
                label=data_record["label"],
                clf_weight=self._strategy_config["clf_weight"],
                field_name=field_name,
                data_record=data_record,
                ppl_metric=self._ppl_metric,
                ppl_smoothing=self._strategy_config["ppl_smoothing"])

            batch_tensor[:, kk:kk_ed] = torch.tensor(final_idxs).to(self._device)

        batch_tensor = batch_tensor.detach().cpu().numpy()
        return [tostring(self._tokenizer, x[1:-1]) for x in batch_tensor]

    def paraphrase_example(self, data_record, field_name, n):
        global STATS_TOTAL
        global STATS_ACCEPT

        if self._strategy_config["lm_option"] == "adv":
            self._bert_lm = self._bert_lms[data_record["label"]]
            self._bert_lm.to(self._device)

        clipped_text = " ".join(data_record[field_name].split()[:200])
        clipped_text = process_text(clipped_text, PRE_PROCESSING_PATTERN)
        batch_size = self._strategy_config["batch_size"]

        sentences = []
        n_batches = math.ceil(n / batch_size)
        last_batch_size = n % batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size

        for id in range(n_batches):
            current_burnin_steps = self._strategy_config["burnin_steps"]
            current_sampling_steps = self._strategy_config["sampling_steps"]

            if self._strategy_config["split_sentence"] == "0":
                batch = self._parallel_sequential_generation(
                    clipped_text,
                    batch_size if id != n_batches - 1 else last_batch_size,
                    current_burnin_steps,
                    current_sampling_steps,
                    field_name, data_record)
                sentences += batch
            elif self._strategy_config["split_sentence"] in ["1", "auto"]:
                splitted_text_ori = self._text_parser.split_paragraph_to_sentences(
                    clipped_text)

                if self._strategy_config["split_sentence"] == "auto":
                    splitted_text = []
                    current_text = ""
                    for s in splitted_text_ori:
                        current_text += " " + s
                        if len(current_text.split()) > AUTO_SENTENCE_LEN_THRESHOLD:
                            splitted_text.append(current_text)
                            current_text = ""
                    if len(current_text.split()) > 0:
                        splitted_text.append(current_text)

                    current_burnin_steps = self._strategy_config["burnin_steps"] // len(splitted_text)
                    current_sampling_steps = self._strategy_config["sampling_steps"] // len(splitted_text)
                elif self._strategy_config["split_sentence"] == "1":
                    splitted_text = splitted_text_ori
                else:
                    assert 0

                batch_res = [""] * (batch_size if id != n_batches - 1 else last_batch_size)
                for text in splitted_text:
                    if len(text.split()) < self._strategy_config["sentence_len_threshold"]:
                        batch_res = [x + " " + text for x in batch_res]
                        continue

                    batch = self._parallel_sequential_generation(
                        text, batch_size if id != n_batches - 1 else last_batch_size,
                        current_burnin_steps,
                        current_sampling_steps,
                        field_name, data_record)

                    batch_res = [x + " " + y for (x, y) in zip(batch_res, batch)]
                sentences += batch_res
            else:
                assert 0

        assert len(sentences) == n

        if self._strategy_config["lm_option"] == "adv":
            self._bert_lm.to(torch.device("cpu"))

        if STATS_TOTAL > 0:
            logger.info("Aggregated accept rate: %.2lf%%.", STATS_ACCEPT / STATS_TOTAL * 100)
        return sentences[:n]
