import math

import numpy as np
import torch

from fibber import log
from fibber.metrics.bert_lm_utils import get_lm
from fibber.paraphrase_strategies.asrs_strategy import all_accept_criteria, tostring
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


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
    if sim_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")
    use_semantic_similarity = sim_metric.measure_batch(origin, paraphrases)
    return -sim_weight * (
        np.maximum(sim_threshold - np.asarray(use_semantic_similarity), 0) ** 2)


def ppl_criteria_score(origin, paraphrases, ppl_metric, ppl_weight, seed_ppl):
    """Estimate the score of a sentence using USE.

    Args:
        origin (str): original sentence.
        paraphrases ([str]): a list of paraphrase_list.
        ppl_metric (GPT2PerplexityMetric): a GPT2PerplexityMetric metric object.
        ppl_weight (float): the weight parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    if ppl_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")
    ppl = ppl_metric.measure_batch(origin, paraphrases)
    # ppl_ratio = np.asarray(ppl_ratio) / np.asarray(seed_ppl)
    return -ppl_weight * np.maximum(np.asarray(ppl) - 20, 0)


def clf_criteria_score(origin, paraphrases, data_record, field_name, clf_metric, clf_weight):
    if clf_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")

    dist = clf_metric.predict_log_dist_batch(origin, paraphrases, data_record, field_name)
    label = data_record["label"]
    # correct_prob = (dist[:, label]).copy()
    dist[:, label] = -1e8
    incorrect_prob = np.max(dist, axis=1)
    return -clf_weight * np.maximum(1 - incorrect_prob, 0)


def bleu_criteria_score(origin, paraphrases, bleu_metric, bleu_weight, bleu_threshold):
    if bleu_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")
    bleu_score = bleu_metric.measure_batch(origin, paraphrases)
    return -bleu_weight * np.maximum(bleu_threshold - np.asarray(bleu_score), 0)


def joint_weighted_criteria(
        tokenizer, data_record, field_name, origin, previous_sents, candidate_sents,
        sim_metric, sim_threshold, sim_weight, clf_metric, clf_weight, ppl_metric, ppl_weight,
        burnin_weight, stats, decision_fn_state, seed_ppl, bleu_weight, bleu_metric, bleu_threshold):
    """Accept or reject candidate word using the joint weighted criteria.

    Returns:
        (np.array, np.array)
            a 2-D int array of size ``batch_size, pos_ed - pos_st``. Each row ``i`` is
                either ``previous_ids[i, :]`` if rejected, or ``candidate_ids[i, :]`` if accepted.
            a 1-D float array of criteria score.
    """
    if burnin_weight == 0:
        return previous_sents

    def compute_criteria_score(paraphrases):
        ppl_score = ppl_criteria_score(
            origin=origin, paraphrases=paraphrases, ppl_metric=ppl_metric, ppl_weight=ppl_weight,
            seed_ppl=seed_ppl)
        sim_score = sim_criteria_score(
            origin=origin, paraphrases=paraphrases, sim_metric=sim_metric, sim_weight=sim_weight,
            sim_threshold=sim_threshold)
        clf_score = clf_criteria_score(origin=origin, paraphrases=paraphrases,
                                       data_record=data_record, field_name=field_name,
                                       clf_metric=clf_metric,
                                       clf_weight=clf_weight)
        bleu_score = bleu_criteria_score(origin=origin, paraphrases=paraphrases,
                                         bleu_weight=bleu_weight, bleu_metric=bleu_metric,
                                         bleu_threshold=bleu_threshold)
        # print("ppl", np.mean(ppl_score), np.std(ppl_score))
        # print("sim", np.mean(sim_score), np.std(sim_score))
        # print("clf", np.mean(clf_score), np.std(clf_score))
        return ppl_score + sim_score + clf_score + bleu_score

    if decision_fn_state is not None:
        previous_criteria_score = decision_fn_state
    else:
        previous_criteria_score = compute_criteria_score(previous_sents)

    candidate_criteria_score = compute_criteria_score(candidate_sents)

    alpha = np.exp((candidate_criteria_score - previous_criteria_score) * burnin_weight)

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")

    stats["accept"] += np.sum(accept)
    stats["all"] += len(accept)
    state = candidate_criteria_score * accept + previous_criteria_score * (1 - accept)

    res = [cand if acc else prev
           for prev, cand, acc in zip(previous_sents, candidate_sents, accept)]
    return res, state


class SSRSStrategy(StrategyBase):
    __abbr__ = "ssrs"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 3, "the block sampling window size."),
        ("burnin_steps", int, 100, "number of burnin steps."),
        ("sampling_steps", int, 200, "number of sampling steps (including) burnin."),
        ("sim_threshold", float, 0.95, "the threshold for USE similarity."),
        ("sim_weight", float, 500, "the smoothing parameter for USE similarity."),
        ("accept_criteria", str, "joint_weighted_criteria", (
            "accept criteria for candidate words [all, joint_weighted_criteria].")),
        ("burnin_criteria_schedule", str, "1", ("the schedule decides how strict the criteria is "
                                                "used. options are [linear, 0, 1].")),
        ("clf_weight", float, 3, "weight for the clf score in the criteria."),
        ("ppl_weight", float, 5, "the smoothing parameter for gpt2."),
        ("sim_metric", str, "CESimilarityMetric", "similarity metric"),
        ("bleu_weight", float, 0, "the weight for self-bleu metric."),
        ("bleu_threshold", float, 1.0, "the weight for self-bleu metric."),
    ]

    _bert_lms = None
    _sim_metric = None
    _clf_metric = None
    _ppl_metrics = None
    _decision_fn = None
    _enforcing_dist_fn = None
    _stats = None
    _tokenizer = None
    _bleu_metric = None

    def __repr__(self):
        return self.__class__.__name__ + "-" + self._strategy_config["sim_metric"]

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for ASRSStrategy.")

        self._tokenizer, self._bert_lms = get_lm("adv", self._dataset_name, trainset, self._device)

        # Load useful metrics
        self._sim_metric = self._metric_bundle.get_metric(
            self._strategy_config["sim_metric"])
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._ppl_metrics = [
            self._metric_bundle.get_metric("BertPerplexityMetric-exclude-%s" % label)
            for label in trainset["label_mapping"]]
        self._bleu_metric = self._metric_bundle.get_metric("SelfBleuMetric")

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

    def _parallel_sequential_generation(self, original_text, seeds, batch_size, burnin_steps,
                                        sampling_steps, field_name, data_record, bert_lm):
        if len(seeds) == 0:
            seeds = [original_text]

        previous_sents = []
        for i in range(batch_size):
            previous_sents.append(np.random.choice(seeds))

        decision_fn_state = None

        seed_ppl = self._ppl_metrics[data_record["label"]].measure_batch(
            original_text, previous_sents)

        for ii in range(sampling_steps):
            batch_input = self._tokenizer(text=previous_sents, padding=True)
            input_ids = torch.tensor(batch_input["input_ids"])
            attention_mask = torch.tensor(batch_input["attention_mask"])
            del batch_input

            expanded_size = list(input_ids.size())
            expanded_size[1] += 1
            input_ids_tmp = torch.zeros(size=expanded_size, dtype=torch.long)
            attention_mask_tmp = torch.zeros(size=expanded_size, dtype=torch.long)
            actual_len_list = attention_mask.sum(dim=1).detach().cpu().numpy()

            op_info = []
            half_window = self._strategy_config["window_size"] // 2

            for j, actual_len in enumerate(actual_len_list):
                if actual_len < 3:
                    op = "I"
                else:
                    op = np.random.choice(["I", "D", "R"])

                # p w_st w_ed is always the index in the expanded (new) tensor, inclusive
                if op == "I":
                    p = np.random.randint(1, actual_len)
                    w_st = max(p - half_window, 1)
                    w_ed = min(p + half_window + 1, actual_len - 1)
                    input_ids_tmp[j, :p] = input_ids[j, :p]
                    input_ids_tmp[j, p + 1:] = input_ids[j, p:]
                    input_ids_tmp[j, w_st:w_ed + 1] = self._tokenizer.mask_token_id
                    attention_mask_tmp[j, :actual_len + 1] = 1
                    op_info.append((op, p, w_st, w_ed))
                elif op == "D":
                    p = np.random.randint(1, actual_len - 1)
                    w_st = max(p - half_window, 1)
                    w_ed = min(p + half_window, actual_len - 3)
                    input_ids_tmp[j, :p] = input_ids[j, :p]
                    input_ids_tmp[j, p:-2] = input_ids[j, p + 1:]
                    input_ids_tmp[j, w_st:w_ed + 1] = self._tokenizer.mask_token_id
                    attention_mask_tmp[j, :actual_len - 1] = 1
                    op_info.append((op, p, w_st, w_ed))
                elif op == "R":
                    p = np.random.randint(1, actual_len - 1)
                    w_st = max(p - half_window, 1)
                    w_ed = min(p + half_window, actual_len - 2)
                    input_ids_tmp[j, :-1] = input_ids[j, :]
                    input_ids_tmp[j, w_st:w_ed + 1] = self._tokenizer.mask_token_id
                    attention_mask_tmp[j, :actual_len] = 1
                    op_info.append((op, p, w_st, w_ed))
                else:
                    assert 0

            logits_lm = bert_lm(input_ids_tmp.to(self._device),
                                attention_mask=attention_mask_tmp.to(self._device))[0]
            logits_lm[:, :, self._tokenizer.sep_token_id] = -1e8

            input_ids_tmp = logits_lm.argmax(dim=2).detach().cpu().numpy()
            del logits_lm

            candidate_sents = []

            for j in range(batch_size):
                op, p, w_st, w_ed = op_info[j]
                if op == "I":
                    toks = (list(input_ids[j, 1:w_st])
                            + list(input_ids_tmp[j, w_st:w_ed + 1])
                            + list(input_ids[j, w_ed:actual_len_list[j] - 1]))
                elif op == "D":
                    toks = (list(input_ids[j, 1:w_st])
                            + list(input_ids_tmp[j, w_st:w_ed + 1])
                            + list(input_ids[j, w_ed + 2:actual_len_list[j] - 1]))
                elif op == "R":
                    toks = (list(input_ids[j, 1:w_st])
                            + list(input_ids_tmp[j, w_st:w_ed + 1])
                            + list(input_ids[j, w_ed + 1:actual_len_list[j] - 1]))
                else:
                    assert 0
                candidate_sents.append(tostring(self._tokenizer, toks))

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

            previous_sents, decision_fn_state = self._decision_fn(
                tokenizer=self._tokenizer, data_record=data_record, field_name=field_name,
                origin=original_text, previous_sents=previous_sents,
                candidate_sents=candidate_sents,
                sim_metric=self._sim_metric, sim_threshold=self._strategy_config["sim_threshold"],
                sim_weight=self._strategy_config["sim_weight"], clf_metric=self._clf_metric,
                clf_weight=self._strategy_config["clf_weight"],
                ppl_metric=self._ppl_metrics[data_record["label"]],
                ppl_weight=self._strategy_config["ppl_weight"],
                burnin_weight=decision_fn_burnin_weight, stats=self._stats,
                decision_fn_state=decision_fn_state,
                seed_ppl=seed_ppl,
                bleu_weight=self._strategy_config["bleu_weight"],
                bleu_metric=self._bleu_metric,
                bleu_threshold=self._strategy_config["bleu_threshold"])

        return previous_sents

    def paraphrase_example(self, data_record, field_name, n):
        clipped_text = data_record[field_name]
        batch_size = self._strategy_config["batch_size"]

        sentences = []
        n_batches = math.ceil(n / batch_size)
        last_batch_size = n % batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size

        bert_lm = self._bert_lms[data_record["label"]].to(self._device)

        for idx in range(n_batches):
            burnin_steps = self._strategy_config["burnin_steps"]
            sampling_steps = self._strategy_config["sampling_steps"]

            if "seeds" in data_record:
                seeds = data_record["seeds"] + [clipped_text] * max(1, len(data_record["seeds"]))
            else:
                seeds = [clipped_text]
            with torch.no_grad():
                batch = self._parallel_sequential_generation(
                    clipped_text,
                    seeds,
                    batch_size if idx != n_batches - 1 else last_batch_size,
                    burnin_steps, sampling_steps, field_name, data_record,
                    bert_lm=bert_lm)
                sentences += batch
        bert_lm.cpu()

        assert len(sentences) == n

        logger.info("Aggregated accept rate: %.2lf%%.",
                    self._stats["accept"] / self._stats["all"] * 100)
        return sentences[:n]
