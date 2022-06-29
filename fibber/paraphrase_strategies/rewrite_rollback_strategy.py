
import numpy as np
import torch
from nltk import word_tokenize
from torch import nn
from torch.nn import functional as F

from fibber import log
from fibber.metrics.bert_lm_utils import get_lm
from fibber.paraphrase_strategies.asrs_utils_wpe import get_wordpiece_emb
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)

REDFLAG_WORDS = []


def roll_back(record, adv, clf_metric):
    s1 = word_tokenize(record["text0"])
    s2 = word_tokenize(adv)

    f = np.zeros((len(s1) + 1, len(s2) + 1), dtype='int')

    for i in range(len(s1) + 1):
        f[i, 0] = i

    for i in range(len(s2) + 1):
        f[0, i] = i

    for i in range(len(s1)):
        for j in range(len(s2)):
            f[i + 1][j + 1] = min(f[i, j + 1] + 1, f[i + 1, j] + 1, f[i, j] + 1)
            if s1[i] == s2[j]:
                f[i + 1][j + 1] = min(f[i + 1][j + 1], f[i][j])

    p = len(s1)
    q = len(s2)

    def check(toks_prev, toks, record):
        s = " ".join(toks)
        # return clf_metric.predict_example(None, s) != record["label"]
        s_prev = " ".join(toks_prev)
        log_prob = clf_metric.predict_log_dist_multiple_examples(None, [s, s_prev])
        label = record["label"]
        return np.argmax(log_prob[0]) != label or log_prob[0, label] < log_prob[1, label]

    cc = 0

    while p > 0 or q > 0:
        if p == 0:
            s2tmp = s2[:]
            del s2tmp[q - 1]
            if check(s2, s2tmp, record):
                s2 = s2tmp
                cc += 1
            q -= 1
            continue

        if q == 0:
            s2tmp = [s1[p - 1]] + s2[:]
            if check(s2, s2tmp, record):
                s2 = s2tmp
                cc += 1
            p -= 1
            continue

        if s1[p - 1] == s2[q - 1] and f[p][q] == f[p - 1][q - 1]:
            p -= 1
            q -= 1
            continue

        if f[p][q] == f[p][q - 1] + 1:
            s2tmp = s2[:]
            del s2tmp[q - 1]
            if check(s2, s2tmp, record):
                s2 = s2tmp
                cc += 1
            q -= 1
            continue

        if f[p][q] == f[p - 1][q - 1] + 1:
            s2tmp = s2[:]
            s2tmp[q - 1] = s1[p - 1]
            if check(s2, s2tmp, record):
                s2 = s2tmp
                cc += 1
            p -= 1
            q -= 1
            continue

        if f[p][q] == f[p - 1][q] + 1:
            s2tmp = s2[:q] + [s1[p - 1]] + s2[q:]
            if check(s2, s2tmp, record):
                s2 = s2tmp
                cc += 1
            p -= 1
            continue

        logger.warning("incorrect rollback")
        break
        # print(p, q)
        # print(f[p][q], f[p - 1][q], f[p][q - 1], f[p - 1][q - 1])
        # print(s1[p - 1], s2[q - 1])
        # assert 0

    return " ".join(s2)


def tostring(tokenizer, seq):
    """Convert a sequence of word ids to a sentence. The post prossing is applied.

    Args:
        tokenizer (transformers.BertTokenizer): a BERT tokenizer.
        seq (list): a list-like sequence of word ids.
    """
    return tokenizer.decode(seq)


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


def all_accept_criteria(candidate_paraphrases, **kwargs):
    """Always accept proposed words.
    """
    return candidate_paraphrases, None


def sim_criteria_score(origin_list, paraphrases, sim_metric, sim_threshold, sim_weight):
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
    use_semantic_similarity = sim_metric.measure_multiple_examples(origin_list, paraphrases)
    return (-sim_weight * (
        np.maximum(sim_threshold - np.asarray(use_semantic_similarity), 0)),
        use_semantic_similarity)


def ppl_criteria_score(origin_list, paraphrases, ppl_metric, ppl_weight):
    """Estimate the score of a sentence using USE.

    Args:
        origin (str): original sentence.
        paraphrases ([str]): a list of paraphrase_list.
        ppl_metric (GPT2PerplexityMetric): a GPT2PerplexityMetric metric object.
        ppl_weight (float): the weight parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    ppl_ratio = ppl_metric.measure_multiple_examples(origin_list, paraphrases, use_ratio=True)
    return (-ppl_weight * (np.maximum(np.asarray(ppl_ratio) - 1, 0)),
            ppl_ratio)


def clf_criteria_score(origin_list, paraphrases, data_record_list, field, clf_metric,
                       clf_weight):
    if clf_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")

    dist_list = clf_metric.predict_log_dist_multiple_examples(
        origin_list, paraphrases, data_record_list)
    # dist_list = np.exp(dist_list)

    scores = []
    not_correct = []
    for pred_dist, data_record in zip(dist_list, data_record_list):
        label = data_record["label"]
        correct_prob = (pred_dist[label]).copy()
        pred_dist[label] = -1e8
        incorrect_prob = np.max(pred_dist)
        not_correct.append(correct_prob < incorrect_prob)
        # margin = 1 / len(pred_dist)
        scores.append(correct_prob - incorrect_prob)
        # scores.append(1 - incorrect_prob)

    scores = np.asarray(scores)

    return -clf_weight * np.maximum(scores, 0), np.asarray(not_correct)


def joint_weighted_criteria(
        origin_list, prev_paraphrases, candidate_paraphrases,
        data_record_list, field, sim_metric, sim_threshold, sim_weight,
        clf_metric, clf_weight, ppl_metric, ppl_weight, stats, state,
        log_prob_trans_forward, log_prob_trans_backward, edit_metric,
        masked_part_text, filled_in_text, **kwargs):

    def compute_criteria_score(paraphrases):
        ppl_score, ppl_ratio = ppl_criteria_score(origin_list=origin_list, paraphrases=paraphrases,
                                                  ppl_metric=ppl_metric, ppl_weight=ppl_weight)
        sim_score, sim_value = sim_criteria_score(origin_list=origin_list, paraphrases=paraphrases,
                                                  sim_metric=sim_metric, sim_weight=sim_weight,
                                                  sim_threshold=sim_threshold)
        clf_score, is_incorrect = clf_criteria_score(origin_list=origin_list,
                                                     paraphrases=paraphrases,
                                                     data_record_list=data_record_list,
                                                     field=field, clf_metric=clf_metric,
                                                     clf_weight=clf_weight)
        return ppl_score + sim_score + clf_score, is_incorrect

    if state is not None:
        previous_criteria_score = state[0]
        previous_is_incorrect = state[1]
    else:
        previous_criteria_score, previous_is_incorrect = compute_criteria_score(prev_paraphrases)

    candidate_criteria_score, candidate_is_incorrect = compute_criteria_score(
        candidate_paraphrases)

    candidate_criteria_score -= previous_is_incorrect * (1 - candidate_is_incorrect) * 1e8

    alpha = np.exp((candidate_criteria_score - previous_criteria_score
                    + log_prob_trans_backward - log_prob_trans_forward))

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")

    stats["accept"] += np.sum(accept)
    stats["all"] += len(accept)
    state = (candidate_criteria_score * accept + previous_criteria_score * (1 - accept),
             candidate_is_incorrect * accept + previous_is_incorrect * (1 - accept))

    ret = [candidate_paraphrases[idx] if is_acc else prev_paraphrases[idx]
           for idx, is_acc in enumerate(accept)]
    stats["success"] = np.sum(state[1])
    return ret, state


def none_constraint(**kwargs):
    return 0.


def compute_wpe_embedding(sents, word_embs, tokenizer, device):
    batch_input = tokenizer(sents, padding=True, return_tensors="pt").to(device)
    return (word_embs(batch_input.input_ids) * batch_input.attention_mask[:, :, None]).sum(dim=1)


def wpe_constraint(target_emb, word_embs, paraphrases_with_mask, n_masks, tokenizer,
                   wpe_threshold, wpe_weight, device, **kwargs):
    current_emb = compute_wpe_embedding(paraphrases_with_mask, word_embs, tokenizer, device)
    dis = F.cosine_similarity((target_emb - current_emb)[:, None, :],
                              word_embs.weight.data[None, :, :], dim=2)
    dis = (wpe_threshold - dis).clamp_(min=0)
    dis = -wpe_weight * dis
    ret = []
    for idx, cc in enumerate(n_masks):
        ret += [dis[idx]] * cc
    return torch.stack(ret, dim=0)


def count_mask(data, tokenizer):
    counter = 0
    for line in data:
        for tok in tokenizer.tokenize(line):
            if tok == "[MASK]":
                counter += 1
    return counter


def assign_candidates(paraphrases_with_mask, candidate_words, tokenizer, masked_index):
    ret = []
    filled_in_part = []
    p = 0
    for idx, paraphrase in enumerate(paraphrases_with_mask):
        tokens = tokenizer.tokenize(paraphrase)
        while True:
            try:
                mask_index = tokens.index("[MASK]")
            except BaseException:
                ret.append(tokenizer.convert_tokens_to_string(tokens))
                filled_in_part.append(tokenizer.convert_tokens_to_string(
                    tokens[masked_index[idx][0]:masked_index[idx][1]]))
                break
            tokens[mask_index] = candidate_words[p]
            p += 1
    return ret, filled_in_part


def smart_mask(toks_raw, st, ed, op):
    toks = toks_raw[st:ed]
    if op == -1:
        idx = np.random.randint(len(toks))
        del toks[idx]

    ret = []
    counter = 0
    for tok in toks:
        if tok.isalpha() and (tok.lower() == tok) and (tok.lower() not in REDFLAG_WORDS):
            ret.append("[MASK]")
            counter += 1
        else:
            ret.append(tok)

    if op == 1:
        ret += ["[MASK]"]
        counter += op
    return ret, counter


class RewriteRollbackStrategy(StrategyBase):
    __abbr__ = "rr"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 3, "the block sampling window size."),
        ("sampling_steps", int, 200, "number of sampling steps."),
        ("top_k", int, 100, "sample from top k words. Use 0 for all words."),
        ("temperature", float, 1., "the softmax temperature for sampling."),
        ("sim_threshold", float, 0.95, "the threshold for USE similarity."),
        ("sim_weight", float, 500, "the smoothing parameter for USE similarity."),
        ("wpe_threshold", float, 1.00, "the threshold for USE similarity."),
        ("wpe_weight", float, 1000, "the smoothing parameter for USE similarity."),
        ("accept_criteria", str, "joint_weighted_criteria", (
            "accept criteria for candidate words from "
            "[all, joint_weighted_criteria].")),
        ("enforcing_dist", str, "wpe", ("enforcing constraint for candidate "
                                        "words from [none, wpe].")),
        ("lm_option", str, "finetune", "choose from [pretrain, finetune]."),
        ("lm_steps", int, 5000, "lm training steps."),
        ("clf_weight", float, 3, "weight for the clf score in the criteria."),
        ("ppl_weight", float, 5, "the smoothing parameter for gpt2."),
        ("sim_metric", str, "CESimilarityMetric", "similarity metric"),
        ("early_stop", str, "0", "whether to use early stop. [0, one, half]")
    ]

    def __repr__(self):
        return self.__class__.__name__ + "-" + self._strategy_config["sim_metric"]

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for RewriteRollbackStrategy.")

        self._tokenizer, lm = get_lm(
            self._strategy_config["lm_option"], self._dataset_name, trainset, self._device,
            lm_steps=self._strategy_config["lm_steps"])

        self._bert_lm = lm
        self._bert_lm.to(self._device)

        # Load useful metrics
        self._sim_metric = self._metric_bundle.get_metric(
            self._strategy_config["sim_metric"])
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._ppl_metric = self._metric_bundle.get_metric("BertPerplexityMetric")
        self._edit_metric = self._metric_bundle.get_metric("EditDistanceMetric")

        # load word piece embeddings.
        wpe = get_wordpiece_emb(self._dataset_name, trainset, self._tokenizer, self._device)
        assert wpe.shape[0] == 300
        wpe[:, self._tokenizer.mask_token_id] = 0
        wpe[:, self._tokenizer.sep_token_id] = 0
        wpe[:, self._tokenizer.cls_token_id] = 0

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
        else:
            assert 0

        self._redflag_vocab = np.zeros(self._tokenizer.vocab_size)
        self._redflag_vocab[self._tokenizer.sep_token_id] = 1
        self._redflag_vocab[self._tokenizer.cls_token_id] = 1
        self._redflag_vocab[self._tokenizer.mask_token_id] = 1
        for item in self._tokenizer.convert_tokens_to_ids(REDFLAG_WORDS):
            self._redflag_vocab[item] = 1
        self._redflag_vocab = torch.tensor(self._redflag_vocab).to(self._device)

        self._stats = {
            "all": 0,
            "accept": 0
        }

    def paraphrase_example(self, data_record, n):
        return self.paraphrase_multiple_examples([data_record] * n), 0

    def paraphrase_multiple_examples(self, data_record_list):
        origin = [item[self._field] for item in data_record_list]
        paraphrases = origin[:]
        context = None if self._field == "text0" else [item["text0"] for item in data_record_list]

        sampling_steps = self._strategy_config["sampling_steps"]
        window_size = self._strategy_config["window_size"]

        target_emb = compute_wpe_embedding(paraphrases, word_embs=self._word_embs,
                                           tokenizer=self._tokenizer, device=self._device)

        decision_fn_state = None

        for ii in range(sampling_steps):
            paraphrases_tokenized = [self._tokenizer.tokenize(sent) for sent in paraphrases]
            paraphrases_with_mask = []
            masked_part_text = []
            masked_index = []
            n_masks = []
            for toks in paraphrases_tokenized:
                st = np.random.randint(max(len(toks) - window_size, 1))
                ed = min(st + window_size, len(toks))
                if st - ed < window_size:
                    choices = [0, 1]
                    p = [0.8, 0.2]
                else:
                    choices = [-1, 0, 1]
                    p = [0.1, 0.8, 0.1]
                op = np.random.choice(choices, p=p)
                masked_part, n_mask_tmp = smart_mask(toks, st, ed, op)
                n_masks.append(n_mask_tmp)
                paraphrases_with_mask.append(
                    self._tokenizer.convert_tokens_to_string(
                        toks[:st] + masked_part + toks[ed:]))
                masked_part_text.append(self._tokenizer.convert_tokens_to_string(toks[st:ed]))
                masked_index.append((st, st + len(masked_part)))

            if self._field == "text1":
                batch_input = self._tokenizer(
                    context, paraphrases_with_mask, padding=True,
                    return_tensors="pt").to(self._device)
            else:
                batch_input = self._tokenizer(
                    paraphrases_with_mask, padding=True, return_tensors="pt").to(self._device)

            logits_lm = self._bert_lm(**batch_input)[0]

            logits_for_masked_toks = torch.masked_select(
                logits_lm, (batch_input.input_ids == self._tokenizer.mask_token_id).unsqueeze(2)
            ).view(-1, logits_lm.size(2))
            logits_for_masked_toks -= 1e8 * self._redflag_vocab

            logits_enforcing = self._enforcing_dist_fn(
                target_emb=target_emb,
                word_embs=self._word_embs,
                paraphrases_with_mask=paraphrases_with_mask,
                n_masks=n_masks,
                tokenizer=self._tokenizer,
                wpe_threshold=self._strategy_config["wpe_threshold"],
                wpe_weight=self._strategy_config["wpe_weight"],
                device=self._device
            )

            logits_joint = logits_for_masked_toks + logits_enforcing
            candidate_ids = sample_word_from_logits(
                logits_joint, top_k=self._strategy_config["top_k"],
                temperature=self._strategy_config["temperature"])

            candidate_paraphrases, filled_in_text = assign_candidates(
                paraphrases_with_mask, self._tokenizer.convert_ids_to_tokens(candidate_ids),
                self._tokenizer, masked_index=masked_index)
            log_prob_previous_ids = 0
            log_prob_candidate_ids = 0

            paraphrases, decision_fn_state = self._decision_fn(
                origin_list=origin, prev_paraphrases=paraphrases,
                candidate_paraphrases=candidate_paraphrases,
                data_record_list=data_record_list,
                field=self._field,
                sim_metric=self._sim_metric,
                sim_threshold=self._strategy_config["sim_threshold"],
                sim_weight=self._strategy_config["sim_weight"],
                clf_metric=self._clf_metric, clf_weight=self._strategy_config["clf_weight"],
                ppl_metric=self._ppl_metric, ppl_weight=self._strategy_config["ppl_weight"],
                stats=self._stats, state=decision_fn_state,
                log_prob_trans_forward=log_prob_candidate_ids,
                log_prob_trans_backward=log_prob_previous_ids,
                edit_metric=self._edit_metric,
                masked_part_text=masked_part_text,
                filled_in_text=filled_in_text)

            if (ii + 1) % 10 == 0:
                for kk in range(len(paraphrases)):
                    if decision_fn_state[1][kk]:
                        paraphrases[kk] = roll_back(
                            data_record_list[kk], paraphrases[kk], self._clf_metric)
                if (self._strategy_config["early_stop"] == "half"
                        and np.sum(decision_fn_state[1]) >= len(data_record_list) * 0.5):
                    break
                if (self._strategy_config["early_stop"] == "5"
                        and np.sum(decision_fn_state[1]) >= 5):
                    break
                decision_fn_state = None

        logger.info("Aggregated accept rate: %.2lf%%. Success rate: %.2lf%%",
                    self._stats["accept"] / self._stats["all"] * 100,
                    self._stats["success"])

        return paraphrases
