import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fibber import log
from fibber.metrics.bert_lm_utils import get_lm
from fibber.paraphrase_strategies.asrs_utils_wpe import get_wordpiece_emb
from fibber.paraphrase_strategies.strategy_base import StrategyBase

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

asrs_clf_counter = 0


def process_text(text, patterns):
    """Processing the text using regex patterns.

    Args:
        text (str): the str to be post processed.
        patterns (list): a list of substitution patterns.
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
    return process_text(tokenizer.decode(seq), POST_PROCESSING_PATTERN)


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


def all_accept_criteria(candidate_ids, stats, **kwargs):
    """Always accept proposed words.

    Args:
        candidate_ids (torch.Tensor): proposed word ids in this sampling step with
            size ``(batch_size, pos_ed-pos_st)``.
        stats (dict): a dict to keep track the accept rate.

    Returns:
        (np.array, None)
            np.array is the same as candidate_ids.
            None means this criteria does not have any state.
    """
    stats["accept"] += len(candidate_ids)
    stats["all"] += len(candidate_ids)
    return candidate_ids, None


def sim_criteria_score(origin, paraphrases, sim_metric, sim_threshold, sim_weight):
    """Estimate the score of a sentence using USE.

    Args:
        origin (str): original sentence.
        paraphrases ([str]): a list of paraphrase_list.
        sim_metric (MetricBase): a similarity metric object.
        sim_threshold (float): the universal sentence encoder similarity threshold.
        sim_weight (float): the weight parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    use_semantic_similarity = sim_metric.measure_batch(origin, paraphrases)
    return (-sim_weight * (
        np.maximum(sim_threshold - np.asarray(use_semantic_similarity), 0) ** 2),
        use_semantic_similarity)


def ppl_criteria_score(origin, paraphrases, ppl_metric, ppl_weight):
    """Estimate the score of a sentence using USE.

    Args:
        origin (str): original sentence.
        paraphrases ([str]): a list of paraphrase_list.
        ppl_metric (GPT2PerplexityMetric): a GPT2PerplexityMetric metric object.
        ppl_weight (float): the weight parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    ppl_ratio = ppl_metric.measure_batch(origin, paraphrases, use_ratio=True)
    return (-ppl_weight * (np.maximum(np.asarray(ppl_ratio) - 1, 0) ** 2),
            ppl_ratio)


def clf_criteria_score(origin, paraphrases, data_record, field, clf_metric, clf_weight):
    global asrs_clf_counter
    if clf_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")

    dist = clf_metric.predict_log_dist_batch(origin, paraphrases, data_record)
    label = data_record["label"]
    correct_prob = (dist[:, label]).copy()
    dist[:, label] = -1e8
    incorrect_prob = np.max(dist, axis=1)
    asrs_clf_counter += len(paraphrases)
    return (-clf_weight * np.maximum(correct_prob - incorrect_prob, 0),
            incorrect_prob > correct_prob)


def joint_weighted_criteria(
        tokenizer, data_record, field, origin, batch_tensor,
        pos_st, pos_ed, previous_ids, candidate_ids, sim_metric, sim_threshold, sim_weight,
        clf_metric, clf_weight, ppl_metric, ppl_weight, burnin_weight, stats, state,
        device, seq_len, log_prob_previous_ids, log_prob_candidate_ids, **kwargs):
    """Accept or reject candidate word using the joint weighted criteria.

    Args:
        tokenizer (transformers.BertTokenizer): a bert tokenizer.
        data_record (dict): the data record dict.
        field (str): the field to rewritten.
        origin (str): original text. Same as ``data_record[field]``.
        batch_tensor (torch.Tensor): tensor of a batch of text with size ``(batch_size, L)``.
        pos_st (int): the start position of sampling (include).
        pos_ed (int): the end position of sampling (exclude).
        previous_ids (torch.Tensor): word ids before current step of sampling with
            size ``(batch_size, pos_ed-pos_st)``.
        candidate_ids (torch.Tensor): proposed word ids in this sampling step with
            size ``(batch_size, pos_ed-pos_st)``.
        sim_metric (USESimilarityMetric): a universal sentence encoder metric object.
        sim_threshold (float): the universal sentence encoder similarity threshold.
        sim_weight (float): the weight for USE criteria score.
        clf_metric (BertClassifier): a BertClassifier metric.
        clf_weight (float): the weight for BERT criteria score.
        ppl_metric (GPT2PerplexityMetric): a GPT2PerplexityMetric metric.
        ppl_weight (float): the weight for GPT2 criteria score.
        burnin_weight (float): the discount factor.
        stats (dict): a dict to keep track the accept rate.
        state (np.array): the state is criteria score from the previous iteration.
        seq_len (np.array): the valid length for each sentence in the batch.
        device (torch.Device): the device that batch_tensor is on.
    Returns:
        (np.array, np.array)
            a 2-D int array of size ``batch_size, pos_ed - pos_st``. Each row ``i`` is
                either ``previous_ids[i, :]`` if rejected, or ``candidate_ids[i, :]`` if accepted.
            a 1-D float array of criteria score.
    """
    if burnin_weight == 0:
        return all_accept_criteria(candidate_ids, stats)

    def compute_criteria_score(fill_ids):
        batch_tensor[:, pos_st:pos_ed] = fill_ids
        paraphrases = [tostring(tokenizer, x[1:ll - 1])
                       for x, ll in zip(batch_tensor.detach().cpu().numpy(), seq_len)]
        ppl_score, ppl_ratio = ppl_criteria_score(origin=origin, paraphrases=paraphrases,
                                                  ppl_metric=ppl_metric, ppl_weight=ppl_weight)
        sim_score, sim_value = sim_criteria_score(
            origin=origin, paraphrases=paraphrases, sim_metric=sim_metric,
            sim_weight=sim_weight, sim_threshold=sim_threshold)
        clf_score, is_incorrect = clf_criteria_score(
            origin=origin, paraphrases=paraphrases, data_record=data_record,
            field=field, clf_metric=clf_metric, clf_weight=clf_weight)
        return ppl_score + sim_score + clf_score, is_incorrect

    if state is not None:
        previous_criteria_score = state[0]
        previous_is_incorrect = state[1]
    else:
        previous_criteria_score, previous_is_incorrect = compute_criteria_score(previous_ids)

    candidate_criteria_score, candidate_is_incorrect = compute_criteria_score(candidate_ids)

    alpha = np.exp((candidate_criteria_score - previous_criteria_score
                    + log_prob_previous_ids - log_prob_candidate_ids) * burnin_weight)

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")

    stats["accept"] += np.sum(accept)
    stats["all"] += len(accept)
    state = (candidate_criteria_score * accept + previous_criteria_score * (1 - accept),
             candidate_is_incorrect * accept + previous_is_incorrect * (1 - accept))

    accept = torch.tensor(accept).to(device)
    ids = candidate_ids * accept.reshape(-1, 1) + previous_ids * (1 - accept.reshape(-1, 1))
    return ids, state


def none_constraint(**kwargs):
    return 0.


def allow_list_constraint(allow_list, **kwargs):
    return -1e6 * (1 - allow_list)


def wpe_constraint(target_emb, word_embs, batch_tensor, pos_st, pos_ed,
                   wpe_threshold, wpe_weight, attention_mask_paraphrase_text_only, **kwargs):
    current_emb = word_embs(batch_tensor) * attention_mask_paraphrase_text_only[:, :, None]
    current_emb[:, pos_st, pos_ed] = 0
    current_emb = current_emb.sum(dim=1)
    candidate_emb = current_emb[:, None, :] + word_embs.weight.data[None, :, :]
    dis = F.cosine_similarity(candidate_emb, target_emb[:, None, :], dim=2)
    dis = (wpe_threshold - dis).clamp_(min=0)
    return -wpe_weight * dis * dis


class ASRSStrategy(StrategyBase):
    __abbr__ = "asrs"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 3, "the block sampling window size."),
        ("burnin_steps", int, 100, "number of burnin steps."),
        ("sampling_steps", int, 200, "number of sampling steps (including) burnin."),
        ("top_k", int, 100, "sample from top k words after burnin. Use 0 for all words."),
        ("temperature", float, 1., "the softmax temperature for sampling."),
        ("sim_threshold", float, 0.95, "the threshold for USE similarity."),
        ("sim_weight", float, 500, "the smoothing parameter for USE similarity."),
        ("wpe_threshold", float, 1.00, "the threshold for USE similarity."),
        ("wpe_weight", float, 1000, "the smoothing parameter for USE similarity."),
        ("burnin_enforcing_schedule", str, "1",
            ("the schedule decides how much additional "
             "constraint is added. options are [linear, 0, 1].")),
        ("accept_criteria", str, "joint_weighted_criteria", (
            "choose an accept criteria for candidate words from "
            "[all, joint_weighted_criteria].")),
        ("enforcing_dist", str, "wpe", ("choose an additional constraint for candidate "
                                        "words from [none, allow_list, wpe].")),
        ("burnin_criteria_schedule", str, "1", ("the schedule decides how strict the criteria is "
                                                "used. options are [linear, 0, 1].")),
        ("seed_option", str, "origin", ("the option for seed sentences in generation. "
                                        "choose from [origin, dynamic_len].")),
        ("dynamic_len_min", int, -3, "change length min."),
        ("dynamic_len_max", int, 3, "change length max."),
        ("lm_option", str, "finetune", "choose from [pretrain, finetune, adv]."),
        ("lm_steps", int, 5000, "lm training steps."),
        ("clf_weight", float, 3, "weight for the clf score in the criteria."),
        ("ppl_weight", float, 5, "the smoothing parameter for gpt2."),
        ("sim_metric", str, "USESimilarityMetric", "similarity metric")
    ]

    def __repr__(self):
        return self.__class__.__name__ + "-" + self._strategy_config["sim_metric"]

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for ASRSStrategy.")

        self._tokenizer, lm = get_lm(
            self._strategy_config["lm_option"], self._dataset_name, trainset, self._device,
            lm_steps=self._strategy_config["lm_steps"])

        if isinstance(lm, list):
            self._bert_lms = lm
        else:
            self._bert_lm = lm
            self._bert_lm.to(self._device)

        # Load useful metrics
        self._sim_metric = self._metric_bundle.get_metric(
            self._strategy_config["sim_metric"])
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._ppl_metric = self._metric_bundle.get_metric("BertPerplexityMetric")

        # load word piece embeddings.
        wpe = get_wordpiece_emb(self._dataset_name, trainset, self._tokenizer, self._device)
        self._word_embs = nn.Embedding(self._tokenizer.vocab_size, 300)
        self._word_embs.weight.data = torch.tensor(wpe.T).float()
        self._word_embs = self._word_embs.to(self._device)
        self._word_embs.eval()
        for item in self._word_embs.parameters():
            item.requires_grad = False

        # config _decision_fn and _enforcing_dist_fn
        if self._strategy_config["accept_criteria"] == "all":
            self._decision_fn = all_accept_criteria
        elif self._strategy_config["accept_criteria"] == "joint_weighted_criteria":
            self._decision_fn = joint_weighted_criteria
        else:
            assert 0

        if self._strategy_config["enforcing_dist"] == "none":
            self._enforcing_dist_fn = none_constraint
        elif self._strategy_config["enforcing_dist"] == "wpe":
            self._enforcing_dist_fn = wpe_constraint
        elif self._strategy_config["enforcing_dist"] == "allow_list":
            self._enforcing_dist_fn = allow_list_constraint
        else:
            assert 0

        self._stats = {
            "all": 0,
            "accept": 0
        }

    def _parallel_sequential_generation(
            self, original_text, seed, batch_size, burnin_steps, sampling_steps, field,
            data_record, early_stop=False):
        if self._strategy_config["seed_option"] == "origin":
            seq = ["[CLS]"] + self._tokenizer.tokenize(seed) + ["[SEP]"]
            batch_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(seq)] * batch_size).to(self._device)
            seq_len = [len(seq)] * batch_size
        elif self._strategy_config["seed_option"] == "dynamic_len":
            seq_raw = self._tokenizer.tokenize(seed)
            seq = []
            seq_len = []
            for i in range(batch_size):
                seq_t = seq_raw[:]
                length_change = np.random.randint(self._strategy_config["dynamic_len_min"],
                                                  self._strategy_config["dynamic_len_max"] + 1)
                # if shrink length, make sure at least 3 words in the sentence.
                if length_change < 0 and len(seq_raw) + length_change < 3:
                    length_change = 3 - len(seq_raw)
                if length_change < 0:
                    for j in range(abs(length_change)):
                        pos = np.random.randint(len(seq_t))
                        seq_t = seq_t[:pos] + seq_t[pos + 1:]
                    seq.append(["[CLS]"] + seq_t + ["[SEP]"])
                    seq_len.append(len(seq_t) + 2)
                else:
                    for j in range(length_change):
                        pos = np.random.randint(len(seq_t))
                        seq_t = seq_t[:pos] + [np.random.choice(seq_raw)] + seq_t[pos:]
                    seq.append(["[CLS]"] + seq_t + ["[SEP]"])
                    seq_len.append(len(seq_t) + 2)
                assert len(seq[-1]) == 2 + len(seq_raw) + length_change
                assert seq_len[-1] == len(seq[-1])
            max_len = max(seq_len)
            seq = [x + ["[PAD]"] * (max_len - len(x)) for x in seq]
            batch_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(x) for x in seq]).to(self._device)
        else:
            assert 0

        attention_mask = torch.zeros_like(batch_tensor)
        attention_mask_paraphrase_text_only = torch.zeros_like(attention_mask)
        for rid, ll in enumerate(seq_len):
            attention_mask[rid, :ll] = 1
            attention_mask_paraphrase_text_only[rid, 1:ll - 1] = 1

        if field == "text1":
            context_seq = ["[CLS]"] + self._tokenizer.tokenize(data_record["text0"])
            context_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(context_seq)] * batch_size
            ).to(self._device)
            context_len = len(context_seq)
            batch_tensor[:, 0] = self._tokenizer.sep_token_id
            attention_mask = torch.cat([
                torch.ones_like(context_tensor),
                attention_mask], dim=1)
        else:
            context_tensor = None

        target_emb = (self._word_embs(torch.tensor([
            self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(original_text))
            for _ in range(batch_size)]).to(self._device))).sum(dim=1)

        allow_list = F.one_hot(batch_tensor[0] * attention_mask_paraphrase_text_only[0],
                               self._tokenizer.vocab_size).sum(dim=0)
        allow_list[0] = 0

        decision_fn_state = None

        for ii in range(sampling_steps):
            pos_st = np.random.randint(1, max(seq_len) - 1)
            pos_ed = min(pos_st + self._strategy_config["window_size"], max(seq_len) - 1)

            previous_ids = batch_tensor[:, pos_st:pos_ed].clone()
            batch_tensor[:, pos_st:pos_ed] = (
                self._tokenizer.mask_token_id
                * attention_mask_paraphrase_text_only[:, pos_st:pos_ed]
                + previous_ids * (1 - attention_mask_paraphrase_text_only[:, pos_st:pos_ed]))

            if field == "text1":
                batch_tensor_tmp = torch.cat([context_tensor, batch_tensor], dim=1)
                tok_type_tensor_tmp = torch.cat([
                    torch.zeros_like(context_tensor),
                    torch.ones_like(batch_tensor)], dim=1)
                logits_lm = self._bert_lm(
                    batch_tensor_tmp,
                    token_type_ids=tok_type_tensor_tmp,
                    attention_mask=attention_mask)[0][
                    :, context_len + pos_st:context_len + pos_ed]
            else:
                logits_lm = self._bert_lm(
                    batch_tensor, attention_mask=attention_mask)[0][:, pos_st:pos_ed]
            logits_lm[:, :, self._tokenizer.sep_token_id] = -1e8

            logits_enforcing = self._enforcing_dist_fn(
                # for wpe constraint
                target_emb=target_emb,
                word_embs=self._word_embs,
                batch_tensor=batch_tensor,
                pos_st=pos_st,
                pos_ed=pos_ed,
                wpe_threshold=self._strategy_config["wpe_threshold"],
                wpe_weight=self._strategy_config["wpe_weight"],
                # for naive constraint
                allow_list=allow_list,
                attention_mask_paraphrase_text_only=attention_mask_paraphrase_text_only
            )

            log_prob_previous_ids = 0
            log_prob_candidate_ids = 0

            sample_order = np.arange(pos_st, pos_ed)
            np.random.shuffle(sample_order)
            for pos in sample_order:
                if ii < burnin_steps:
                    if self._strategy_config["burnin_enforcing_schedule"] == "0":
                        logits_joint = logits_lm[:, pos - pos_st]
                    elif self._strategy_config["burnin_enforcing_schedule"] == "1":
                        logits_joint = logits_lm[:, pos - pos_st] + logits_enforcing
                    elif self._strategy_config["burnin_enforcing_schedule"] == "linear":
                        logits_joint = logits_lm[:, pos - pos_st] + (
                            (ii + 1) / burnin_steps) * logits_enforcing
                    else:
                        assert 0
                else:
                    logits_joint = logits_lm[:, pos - pos_st] + logits_enforcing

                logits_joint = F.log_softmax(logits_joint, dim=1)

                top_k = self._strategy_config["top_k"] if (ii >= burnin_steps) else 0

                candidate_ids = sample_word_from_logits(
                    logits_joint, top_k=top_k, temperature=self._strategy_config["temperature"])

                log_prob_previous_ids = log_prob_previous_ids + (
                    torch.gather(
                        logits_joint, dim=1, index=previous_ids[:, pos - pos_st].unsqueeze(1)
                    ).squeeze(1) * attention_mask_paraphrase_text_only[:, pos])
                log_prob_candidate_ids = log_prob_candidate_ids + (
                    torch.gather(
                        logits_joint, dim=1, index=candidate_ids.unsqueeze(1)).squeeze(1)
                    * attention_mask_paraphrase_text_only[:, pos])

                batch_tensor[:, pos] = (
                    candidate_ids * attention_mask_paraphrase_text_only[:, pos]
                    + batch_tensor[:, pos]
                    * (1 - attention_mask_paraphrase_text_only[:, pos]))

            candidate_ids = batch_tensor[:, pos_st:pos_ed].clone()

            if ii < burnin_steps:
                if self._strategy_config["burnin_criteria_schedule"] == "0":
                    decision_fn_burnin_weight = 0
                elif self._strategy_config["burnin_criteria_schedule"] == "1":
                    decision_fn_burnin_weight = 1
                elif self._strategy_config["burnin_criteria_schedule"] == "linear":
                    decision_fn_burnin_weight = (ii + 1) / burnin_steps
                else:
                    assert 0
            else:
                decision_fn_burnin_weight = 1

            final_ids, decision_fn_state = self._decision_fn(
                tokenizer=self._tokenizer, data_record=data_record, field=field,
                origin=data_record[field], batch_tensor=batch_tensor,
                pos_st=pos_st, pos_ed=pos_ed, previous_ids=previous_ids,
                candidate_ids=candidate_ids, sim_metric=self._sim_metric,
                sim_threshold=self._strategy_config["sim_threshold"],
                sim_weight=self._strategy_config["sim_weight"],
                clf_metric=self._clf_metric, clf_weight=self._strategy_config["clf_weight"],
                ppl_metric=self._ppl_metric, ppl_weight=self._strategy_config["ppl_weight"],
                burnin_weight=decision_fn_burnin_weight, stats=self._stats,
                state=decision_fn_state, device=self._device,
                seq_len=seq_len,
                log_prob_candidate_ids=log_prob_candidate_ids.detach().cpu().numpy(),
                log_prob_previous_ids=log_prob_previous_ids.detach().cpu().numpy())

            batch_tensor[:, pos_st:pos_ed] = final_ids
            if early_stop and np.sum(decision_fn_state[1]) >= 3:
                break

        return [tostring(self._tokenizer, x[1:ll - 1])
                for x, ll in zip(batch_tensor.detach().cpu().numpy(), seq_len)]

    def paraphrase_example(self, data_record, n, early_stop=False):
        global asrs_clf_counter
        asrs_clf_counter = 0

        if self._strategy_config["lm_option"] == "adv":
            self._bert_lm = self._bert_lms[data_record["label"]]
            self._bert_lm.to(self._device)

        clipped_text = data_record[self._field]
        clipped_text = process_text(clipped_text, PRE_PROCESSING_PATTERN)
        batch_size = self._strategy_config["batch_size"]

        sentences = []
        n_batches = math.ceil(n / batch_size)
        last_batch_size = n % batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size

        for id in range(n_batches):
            burnin_steps = self._strategy_config["burnin_steps"]
            sampling_steps = self._strategy_config["sampling_steps"]

            batch = self._parallel_sequential_generation(
                clipped_text,
                clipped_text if "seed" not in data_record else data_record["seed"],
                batch_size if id != n_batches - 1 else last_batch_size,
                burnin_steps,
                sampling_steps,
                self._field, data_record,
                early_stop=early_stop)
            sentences += batch

        assert len(sentences) == n

        if self._strategy_config["lm_option"] == "adv":
            self._bert_lm.to(torch.device("cpu"))

        logger.info("Aggregated accept rate: %.2lf%%.",
                    self._stats["accept"] / self._stats["all"] * 100)
        return sentences[:n], asrs_clf_counter
