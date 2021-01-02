import math
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertTokenizerFast

from fibber import log, resources
from fibber.paraphrase_strategies.bert_sampling_utils_lm import get_lm
from fibber.paraphrase_strategies.bert_sampling_utils_text_parser import TextParser
from fibber.paraphrase_strategies.bert_sampling_utils_wpe import get_wordpiece_emb
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

AUTO_SENTENCE_LEN_THRESHOLD = 50  # 50 words


def process_text(text, patterns):
    """Processing the text using regex patterns.

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


def all_accept_criteria(candidate_ids, stats, **kargs):
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


def use_criteria_score(origin, paraphrases, use_metric, use_threshold, use_weight):
    """Estimate the score of a sentence using USE.

    Args:
        origin (str): original sentence.
        paraphrases ([str]): a list of paraphrase_list.
        use_metric (USESemanticSimilarityMetric): a universal sentence encoder metric object.
        use_threshold (float): the universal sentence encoder similarity threshold.
        use_weight (float): the weight parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    if use_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")
    use_semantic_similarity = use_metric.measure_batch(origin, paraphrases)
    return -use_weight * (
        np.maximum(use_threshold - np.asarray(use_semantic_similarity), 0) ** 2)


def gpt2_criteria_score(origin, paraphrases, gpt2_metric, gpt2_weight):
    """Estimate the score of a sentence using USE.

    Args:
        origin (str): original sentence.
        paraphrases ([str]): a list of paraphrase_list.
        gpt2_metric (GPT2GrammarQualityMetric): a GPT2GrammarQualityMetric metric object.
        gpt2_weight (float): the weight parameter for the criteria.

    Returns:
        (np.array): a numpy array of size ``(batch_size,)``. All entries ``<=0``.
    """
    if gpt2_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")
    gpt2_ppl_ratio = gpt2_metric.measure_batch(origin, paraphrases)
    return -gpt2_weight * (np.maximum(gpt2_ppl_ratio, 0) ** 2)


def bert_criteria_score(origin, paraphrases, data_record, field_name, bert_metric, bert_weight):
    if bert_weight == 0:
        return np.zeros(len(paraphrases), dtype="float32")

    dist = bert_metric.predict_dist_batch(origin, paraphrases, data_record, field_name)
    label = data_record["label"]
    correct_prob = (dist[:, label]).copy()
    dist[:, label] = -1e8
    incorrect_prob = np.max(dist, axis=1)
    return -bert_weight * np.maximum(correct_prob - incorrect_prob, 0)


def joint_weighted_criteria(
        tokenizer, data_record, field_name, origin, batch_tensor,
        pos_st, pos_ed, previous_ids, candidate_ids, use_metric, use_threshold, use_weight,
        bert_metric, bert_weight, gpt2_metric, gpt2_weight, burnin_weight, stats, state,
        device, **kargs):
    """Accept or reject candidate word using the joint weighted criteria.

    Args:
        tokenizer (transformers.BertTokenizer): a bert tokenizer.
        data_record (dict): the data record dict.
        field_name (str): the field to rewritten.
        origin (str): original text. Same as ``data_record[field_name]``.
        batch_tensor (torch.Tensor): tensor of a batch of text with size ``(batch_size, L)``.
        pos_st (int): the start position of sampling (include).
        pos_ed (int): the end position of sampling (exclude).
        previous_ids (torch.Tensor): word ids before current step of sampling with
            size ``(batch_size, pos_ed-pos_st)``.
        candidate_ids (torch.Tensor): proposed word ids in this sampling step with
            size ``(batch_size, pos_ed-pos_st)``.
        use_metric (USESemanticSimilarityMetric): a universal sentence encoder metric object.
        use_threshold (float): the universal sentence encoder similarity threshold.
        use_weight (float): the weight for USE criteria score.
        bert_metric (BertClassifier): a BertClassifier metric.
        bert_weight (float): the weight for BERT criteria score.
        gpt2_metric (GPT2GrammarQualityMetric): a GPT2GrammarQualityMetric metric.
        gpt2_weight (float): the weight for GPT2 criteria score.
        burnin_weight (float): the discount factor.
        stats (dict): a dict to keep track the accept rate.
        state (np.array): the state is criteria score from the previous iteration.
        device (torch.Device): the device that batch_tensor is on.
    Returns:
        (np.array, np.array)
            a 2-D int array of size ``batch_size, pos_ed - pos_st``. Each row ``i`` is
                either ``previous_ids[i, :]`` if rejected, or ``candidate_ids[i, :]`` if accepted.
            a 1-D float array of criteria score.
    """
    if burnin_weight == 0:
        return candidate_ids

    def compute_criteria_score(fill_ids):
        batch_tensor[:, pos_st:pos_ed] = fill_ids
        paraphrases = [tostring(tokenizer, x) for x in batch_tensor.detach().cpu().numpy()]
        return (gpt2_criteria_score(origin=origin, paraphrases=paraphrases,
                                    gpt2_metric=gpt2_metric, gpt2_weight=gpt2_weight)
                + use_criteria_score(origin=origin, paraphrases=paraphrases, use_metric=use_metric,
                                     use_weight=use_weight, use_threshold=use_threshold)
                + bert_criteria_score(origin=origin, paraphrases=paraphrases,
                                      data_record=data_record, field_name=field_name,
                                      bert_metric=bert_metric,
                                      bert_weight=bert_weight)) * burnin_weight

    if state is not None:
        previous_criteria_score = state
    else:
        previous_criteria_score = compute_criteria_score(previous_ids)

    candidate_criteria_score = compute_criteria_score(candidate_ids)

    alpha = np.exp(candidate_criteria_score - previous_criteria_score)

    accept = np.asarray(np.random.rand(len(alpha)) < alpha, dtype="int32")

    stats["accept"] += np.sum(accept)
    stats["all"] += len(accept)
    state = candidate_criteria_score * accept + previous_criteria_score * (1 - accept)
    accept = torch.tensor(accept).to(device)
    ids = candidate_ids * accept.reshape(-1, 1) + previous_ids * (1 - accept.reshape(-1, 1))
    return ids, state


def none_constraint(**kargs):
    return 0


def allow_list_constraint(allow_list, **kargs):
    return -1e6 * (1 - allow_list)


def wpe_constraint(target_emb, word_embs, batch_tensor, pos,
                   wpe_threshold, wpe_weight, **kargs):
    current_emb = (word_embs(batch_tensor[:, 1:-1]).sum(dim=1)
                   - word_embs(batch_tensor[:, pos]))
    candidate_emb = current_emb[:, None, :] + word_embs.weight.data[None, :, :]
    dis = F.cosine_similarity(candidate_emb, target_emb[:, None, :], dim=2)
    dis = (wpe_threshold - dis).clamp_(min=0)
    return -wpe_weight * dis * dis


class BertSamplingStrategy(StrategyBase):
    __abbr__ = "bs"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 3, "the block sampling window size."),
        ("burnin_steps", int, 100, "number of burnin steps."),
        ("sampling_steps", int, 200, "number of sampling steps (including) burnin."),
        ("top_k", int, 100, "sample from top k words after burnin. Use 0 for all words."),
        ("temperature", float, 1., "the softmax temperature for sampling."),
        ("use_threshold", float, 0.95, "the threshold for USE similarity."),
        ("use_weight", float, 1000, "the smoothing parameter for USE similarity."),
        ("wpe_threshold", float, 1.00, "the threshold for USE similarity."),
        ("wpe_weight", float, 10000, "the smoothing parameter for USE similarity."),
        ("burnin_enforcing_schedule", str, "1",
            ("the schedule decides how much additional "
             "constraint is added. options are [linear, 0, 1].")),
        ("accept_criteria", str, "joint_weighted_criteria", (
            "select an accept criteria for candidate words from "
            "[all, joint_weighted_criteria].")),
        ("enforcing_dist", str, "wpe", ("select an additional constraint for candidate "
                                        "words from [none, allow_list, wpe].")),
        ("burnin_criteria_schedule", str, "1", ("the schedule decides how strict the criteria is "
                                                "used. options are [linear, 0, 1].")),
        ("seed_option", str, "origin", ("the option for seed sentences in generation. "
                                        "choose from [origin, auto].")),
        ("split_sentence", str, "0", "split paragraph to sentence. options are [0, 1, auto]."),
        ("stanza_port", int, 9000, "stanza port"),
        ("lm_option", str, "finetune", "choose from [pretrain, finetune, adv]."),
        ("lm_steps", int, 5000, "lm training steps."),
        ("clf_weight", float, 1, "weight for the clf score in the criteria."),
        ("gpt2_weight", float, 10, "the smoothing parameter for gpt2."),
    ]

    def __repr__(self):
        return "%s-constraint_%s-criteria_%s" % (
            self.__class__.__name__,
            self._strategy_config["enforcing_dist"],
            self._strategy_config["accept_criteria"])

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for BertSamplingStrategy.")

        if trainset["cased"]:
            model_init = "bert-base-cased"
        else:
            model_init = "bert-base-uncased"
        self._tokenizer = BertTokenizerFast.from_pretrained(
            resources.get_transformers(model_init), do_lower_case="uncased" in model_init)
        self._tokenizer.do_lower_case = True if "uncased" in model_init else False

        if self._strategy_config["lm_option"] == "pretrain":
            self._bert_lm = BertForMaskedLM.from_pretrained(
                resources.get_transformers(model_init)).to(self._device)
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

        # Load useful metrics
        self._use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._gpt2_metric = self._metric_bundle.get_metric("GPT2GrammarQualityMetric")

        # load word piece embeddings.
        wpe = get_wordpiece_emb(self._output_dir, self._dataset_name, trainset, self._device)
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

        # load text parser
        if (self._strategy_config["seed_option"] != "origin"
                or self._strategy_config["split_sentence"] != "0"):
            self._text_parser = TextParser(self._strategy_config["stanza_port"])
        else:
            self._text_parser = None

        self._stats = {
            "all": 0,
            "accept": 0
        }

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

        if field_name == "text1":
            context_seq = ["[CLS]"] + self._tokenizer.tokenize(data_record["text0"])
            context_tensor = torch.tensor(
                [self._tokenizer.convert_tokens_to_ids(context_seq)] * batch_size
            ).to(self._device)
            context_len = len(context_seq)
            batch_tensor[:, 0] = self._tokenizer.sep_token_id
        else:
            context_tensor = None

        target_emb = self._word_embs(batch_tensor[:, 1:-1]).sum(dim=1)

        allow_list = F.one_hot(batch_tensor[0][1:-1], self._tokenizer.vocab_size).sum(dim=0)

        decision_fn_state = None

        for ii in range(sampling_steps):
            pos_st = np.random.randint(1, tensor_len - 1)
            pos_ed = min(pos_st + self._strategy_config["window_size"], tensor_len - 1)

            previous_ids = batch_tensor[:, pos_st:pos_ed].clone()
            batch_tensor[:, pos_st:pos_ed] = self._tokenizer.mask_token_id

            sample_order = np.arange(pos_st, pos_ed)
            np.random.shuffle(sample_order)
            for pos in sample_order:
                if field_name == "text1":
                    batch_tensor_tmp = torch.cat([context_tensor, batch_tensor], dim=1)
                    tok_type_tensor_tmp = torch.cat([
                        torch.zeros_like(context_tensor),
                        torch.ones_like(batch_tensor)], dim=1)
                    logits_lm = self._bert_lm(
                        batch_tensor_tmp,
                        token_type_ids=tok_type_tensor_tmp)[0][:, context_len + pos]
                    logits_lm[:, self._tokenizer.sep_token_id] = -1e8
                else:
                    logits_lm = self._bert_lm(batch_tensor)[0][:, pos]

                logits_enforcing = self._enforcing_dist_fn(
                    # for wpe constraint
                    target_emb=target_emb,
                    word_embs=self._word_embs,
                    batch_tensor=batch_tensor,
                    pos=pos,
                    wpe_threshold=self._strategy_config["wpe_threshold"],
                    wpe_weight=self._strategy_config["wpe_weight"],
                    # for naive constraint
                    allow_list=allow_list
                )

                if ii < burnin_steps:
                    if self._strategy_config["burnin_enforcing_schedule"] == "0":
                        logits_joint = logits_lm
                    elif self._strategy_config["burnin_enforcing_schedule"] == "1":
                        logits_joint = logits_lm + logits_enforcing
                    elif self._strategy_config["burnin_enforcing_schedule"] == "linear":
                        logits_joint = logits_lm + (
                            ii / burnin_steps) * logits_enforcing
                    else:
                        assert 0
                else:
                    logits_joint = logits_lm + logits_enforcing

                top_k = self._strategy_config["top_k"] if (
                    ii >= burnin_steps) else 0

                candidate_ids = sample_word_from_logits(
                    logits_joint, top_k=top_k, temperature=self._strategy_config["temperature"])
                batch_tensor[:, pos] = candidate_ids

            candidate_ids = batch_tensor[:, pos_st:pos_ed].clone()

            if ii < burnin_steps:
                if self._strategy_config["burnin_criteria_schedule"] == "0":
                    decision_fn_burnin_weight = 0
                elif self._strategy_config["burnin_criteria_schedule"] == "1":
                    decision_fn_burnin_weight = 1
                elif self._strategy_config["burnin_criteria_schedule"] == "linear":
                    decision_fn_burnin_weight = ii / burnin_steps
                else:
                    assert 0
            else:
                pass

            final_ids, decision_fn_state = self._decision_fn(
                tokenizer=self._tokenizer, data_record=data_record, field_name=field_name,
                origin=data_record[field_name], batch_tensor=batch_tensor,
                pos_st=pos_st, pos_ed=pos_ed, previous_ids=previous_ids,
                candidate_ids=candidate_ids, use_metric=self._use_metric,
                use_threshold=self._strategy_config["use_threshold"],
                use_weight=self._strategy_config["use_weight"],
                bert_metric=self._clf_metric, bert_weight=self._strategy_config["clf_weight"],
                gpt2_metric=self._gpt2_metric, gpt2_weight=self._strategy_config["gpt2_weight"],
                burnin_weight=decision_fn_burnin_weight, stats=self._stats,
                state=decision_fn_state, device=self._device)

            batch_tensor[:, pos_st:pos_ed] = final_ids

        return [tostring(self._tokenizer, x[1:-1]) for x in batch_tensor.detach().cpu().numpy()]

    def paraphrase_example(self, data_record, field_name, n):
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
            burnin_steps = self._strategy_config["burnin_steps"]
            sampling_steps = self._strategy_config["sampling_steps"]

            if self._strategy_config["split_sentence"] == "0":
                batch = self._parallel_sequential_generation(
                    clipped_text,
                    batch_size if id != n_batches - 1 else last_batch_size,
                    burnin_steps,
                    sampling_steps,
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

                    burnin_steps = self._strategy_config["burnin_steps"] // len(splitted_text)
                    sampling_steps = self._strategy_config["sampling_steps"] // len(splitted_text)
                elif self._strategy_config["split_sentence"] == "1":
                    splitted_text = splitted_text_ori
                    burnin_steps = self._strategy_config["burnin_steps"] // len(splitted_text)
                    sampling_steps = self._strategy_config["sampling_steps"] // len(splitted_text)
                else:
                    assert 0

                batch_res = [""] * (batch_size if id != n_batches - 1 else last_batch_size)
                for text in splitted_text:
                    batch = self._parallel_sequential_generation(
                        text, batch_size if id != n_batches - 1 else last_batch_size,
                        burnin_steps,
                        sampling_steps,
                        field_name, data_record)

                    batch_res = [(x + " " + y).strip() for (x, y) in zip(batch_res, batch)]
                sentences += batch_res
            else:
                assert 0

        assert len(sentences) == n

        if self._strategy_config["lm_option"] == "adv":
            self._bert_lm.to(torch.device("cpu"))

        logger.info("Aggregated accept rate: %.2lf%%.",
                    self._stats["accept"] / self._stats["all"] * 100)
        return sentences[:n]
