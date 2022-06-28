import numpy as np
import torch

from fibber import log
from fibber.metrics.bert_lm_utils import get_lm
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


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
    return (-ppl_weight * (np.maximum(np.asarray(ppl_ratio), 0)),
            ppl_ratio)


def bleu_criteria_score(origin_list, paraphrases, bleu_metric, bleu_weight, bleu_threshold):
    bleu_score = bleu_metric.measure_multiple_examples(origin_list, paraphrases)
    return -bleu_weight * np.maximum(bleu_threshold - np.asarray(bleu_score), 0), bleu_score


def clf_criteria_score(origin_list, paraphrases, data_record_list, field, clf_metric,
                       clf_weight):
    if clf_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")

    dist_list = clf_metric.predict_log_dist_multiple_examples(origin_list, paraphrases,
                                                              data_record_list, field)
    dist_list = np.exp(dist_list)

    scores = []
    not_correct = []
    for pred_dist, data_record in zip(dist_list, data_record_list):
        label = data_record["label"]
        correct_prob = (pred_dist[label]).copy()
        pred_dist[label] = -1e8
        incorrect_prob = np.max(pred_dist)
        not_correct.append(correct_prob < incorrect_prob)
        # margin = 1 / len(pred_dist)
        # scores.append(correct_prob - incorrect_prob)
        scores.append(1 - incorrect_prob)

    scores = np.asarray(scores)

    return -clf_weight * np.maximum(scores, 0), np.asarray(not_correct)


def joint_weighted_criteria(
        origin_list, prev_paraphrases, candidate_paraphrases,
        data_record_list, field, sim_metric, sim_threshold, sim_weight,
        clf_metric, clf_weight, ppl_metric, ppl_weight, stats, state,
        log_prob_trans_forward, log_prob_trans_backward,
        bleu_metric, bleu_weight, bleu_threshold, **kwargs):

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
        bleu_score, bleu_value = bleu_criteria_score(origin_list=origin_list,
                                                     paraphrases=paraphrases,
                                                     bleu_metric=bleu_metric,
                                                     bleu_weight=bleu_weight,
                                                     bleu_threshold=bleu_threshold)
        return ppl_score + sim_score + clf_score + bleu_score, is_incorrect

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
    return ret, state


def none_constraint(**kwargs):
    return 0.


def count_mask(data, tokenizer):
    counter = 0
    for line in data:
        for tok in tokenizer.tokenize(line):
            if tok == "[MASK]":
                counter += 1
    return counter


def assign_cadidates(paraphrases_with_mask, candidate_words, tokenizer):
    ret = []
    p = 0
    for paraphrase in paraphrases_with_mask:
        tokens = tokenizer.tokenize(paraphrase)
        while True:
            try:
                mask_index = tokens.index("[MASK]")
            except BaseException:
                ret.append(tokenizer.convert_tokens_to_string(tokens))
                break
            tokens[mask_index] = candidate_words[p]
            p += 1
    return ret


def smart_mask(toks, op):
    if op == -1:
        idx = np.random.randint(len(toks))
        del toks[idx]

    ret = []
    counter = 0
    for tok in toks:
        ret.append("[MASK]")
        counter += 1

    if op == 1:
        ret += ["[MASK]"]
        counter += op
    return ret, counter


class SSRSStrategy(StrategyBase):
    __abbr__ = "ssrs"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 3, "the block sampling window size."),
        ("sampling_steps", int, 200, "number of sampling steps."),
        ("top_k", int, 100, "sample from top k words. Use 0 for all words."),
        ("temperature", float, 1., "the softmax temperature for sampling."),
        ("sim_threshold", float, 0.95, "the threshold for USE similarity."),
        ("sim_weight", float, 500, "the smoothing parameter for USE similarity."),
        ("accept_criteria", str, "joint_weighted_criteria", (
            "accept criteria for candidate words from "
            "[all, joint_weighted_criteria].")),
        ("lm_option", str, "finetune", "choose from [pretrain, finetune]."),
        ("lm_steps", int, 5000, "lm training steps."),
        ("clf_weight", float, 3, "weight for the clf score in the criteria."),
        ("ppl_weight", float, 5, "the smoothing parameter for gpt2."),
        ("sim_metric", str, "CESimilarityMetric", "similarity metric"),
        ("early_stop", str, "0", "whether to use early stop. [0, one, half]"),
        ("bleu_weight", float, 10, "bleu weight"),
        ("bleu_threshold", float, 0.6, "bleu weight")
    ]

    def __repr__(self):
        return self.__class__.__name__ + "-" + self._strategy_config["sim_metric"]

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for ASRSStrategy.")

        self._tokenizer, self._bert_lms = get_lm("adv", self._dataset_name, trainset, self._device,
                                                 lm_steps=self._strategy_config["lm_steps"])

        # Load useful metrics
        self._sim_metric = self._metric_bundle.get_metric(
            self._strategy_config["sim_metric"])
        self._bleu_metric = self._metric_bundle.get_metric("SelfBleuMetric")
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._ppl_metrics = [
            self._metric_bundle.get_metric("BertPerplexityMetric-exclude-%s" % label)
            for label in trainset["label_mapping"]]

        # config _decision_fn and _enforcing_dist_fn
        if self._strategy_config["accept_criteria"] == "all":
            self._decision_fn = all_accept_criteria
        elif self._strategy_config["accept_criteria"] == "joint_weighted_criteria":
            self._decision_fn = joint_weighted_criteria
        else:
            assert 0

        self._stats = {
            "all": 0,
            "accept": 0
        }

    def paraphrase_example(self, data_record, n):
        return self.paraphrase_multiple_examples([data_record] * n), 0

    def paraphrase_multiple_examples(self, data_record_list):
        bert_lm = self._bert_lms[data_record_list[0]["label"]].to(self._device)

        origin = [item[self._field] for item in data_record_list]
        paraphrases = origin[:]
        context = None if self._field == "text0" else [item["text0"] for item in data_record_list]

        sampling_steps = self._strategy_config["sampling_steps"]
        window_size = self._strategy_config["window_size"]

        decision_fn_state = None
        for ii in range(sampling_steps):
            paraphrases_tokenized = [self._tokenizer.tokenize(sent) for sent in paraphrases]
            paraphrases_with_mask = []
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
                masked_part, n_mask_tmp = smart_mask(toks[st:ed], op)
                n_masks.append(n_mask_tmp)
                paraphrases_with_mask.append(
                    self._tokenizer.convert_tokens_to_string(
                        toks[:st] + masked_part + toks[ed:]))

            if self._field == "text1":
                batch_input = self._tokenizer(
                    context, paraphrases_with_mask, padding=True,
                    return_tensors="pt").to(self._device)
            else:
                batch_input = self._tokenizer(
                    paraphrases_with_mask, padding=True, return_tensors="pt").to(self._device)

            logits_lm = bert_lm(**batch_input)[0]

            logits_for_masked_toks = torch.masked_select(
                logits_lm, (batch_input.input_ids == self._tokenizer.mask_token_id).unsqueeze(2)
            ).view(-1, logits_lm.size(2))
            logits_for_masked_toks[:, self._tokenizer.sep_token_id] = -1e8
            logits_for_masked_toks[:, self._tokenizer.mask_token_id] = -1e8
            logits_for_masked_toks[:, self._tokenizer.cls_token_id] = -1e8

            logits_joint = logits_for_masked_toks
            candidate_ids = sample_word_from_logits(
                logits_joint, top_k=self._strategy_config["top_k"],
                temperature=self._strategy_config["temperature"])

            candidate_paraphrases = assign_cadidates(
                paraphrases_with_mask, self._tokenizer.convert_ids_to_tokens(candidate_ids),
                self._tokenizer)
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
                ppl_metric=self._ppl_metrics[data_record_list[0]["label"]],
                ppl_weight=self._strategy_config["ppl_weight"],
                stats=self._stats, state=decision_fn_state,
                log_prob_trans_forward=log_prob_candidate_ids,
                log_prob_trans_backward=log_prob_previous_ids,
                bleu_metric=self._bleu_metric,
                bleu_weight=self._strategy_config["bleu_weight"],
                bleu_threshold=self._strategy_config["bleu_threshold"])

            if (self._strategy_config["early_stop"] == "half"
                    and np.sum(decision_fn_state[1]) >= len(data_record_list) * 0.5):
                break
            if (self._strategy_config["early_stop"] == "one"
                    and np.sum(decision_fn_state[1]) >= 1):
                break

        logger.info("Aggregated accept rate: %.2lf%%. Success rate: %.2lf%%",
                    self._stats["accept"] / self._stats["all"] * 100,
                    np.sum(decision_fn_state[1]) / len(data_record_list) * 100)

        bert_lm.cpu()

        return paraphrases
