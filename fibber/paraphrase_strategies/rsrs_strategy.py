import math

import numpy as np
import torch

from fibber import log
from fibber.paraphrase_strategies.asrs_strategy import tostring
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


def l2_decision_fn(tokenizer, data_record, field_name,
                   origin, previous_sents, candidate_sents,
                   burnin_weight, decision_fn_state, stats, clf_metric, source_rep):
    def compute_criteria_score(sents):
        seed_batch = clf_metric._tokenizer(
            sents, return_tensors="pt", padding=True).to(clf_metric._device)
        rep = clf_metric._model.bert(**seed_batch)[1]
        err = rep - source_rep
        return -(err * err).sum(dim=1).detach().cpu().numpy()

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


class RSRSStrategy(StrategyBase):
    """Reverse function of sentence embedder."""
    __abbr__ = "rsrs"
    __hyperparameters__ = [
        ("batch_size", int, 50, "the batch size in sampling."),
        ("window_size", int, 3, "the block sampling window size."),
        ("burnin_steps", int, 100, "number of burnin steps."),
        ("sampling_steps", int, 200, "number of sampling steps (including) burnin."),
    ]

    _clf_metric = None

    def __repr__(self):
        return self.__class__.__name__

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load bert language model for ASRSStrategy.")
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._tokenizer = self._clf_metric._tokenizer

    def _parallel_sequential_generation(self, original_text, seeds, batch_size, burnin_steps,
                                        sampling_steps, field_name, data_record):
        if len(seeds) == 0:
            seeds = [original_text]

        previous_sents = []
        for i in range(batch_size):
            previous_sents.append(np.random.choice(seeds))

        decision_fn_state = None

        input_batch = self._tokenizer(original_text, return_tensors="pt").to(self._device)
        source_rep = (self._clf_metric._model.bert(**input_batch)[1]).squeeze(0).data.clone()

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
                    input_ids_tmp[j, p] = self._tokenizer.mask_token_id
                    attention_mask_tmp[j, :actual_len + 1] = 1
                    op_info.append((op, p, w_st, w_ed))
                elif op == "D":
                    p = np.random.randint(1, actual_len - 1)
                    w_st = max(p - half_window, 1)
                    w_ed = min(p + half_window, actual_len - 3)
                    input_ids_tmp[j, :p] = input_ids[j, :p]
                    input_ids_tmp[j, p:-2] = input_ids[j, p + 1:]
                    attention_mask_tmp[j, :actual_len - 1] = 1
                    op_info.append((op, p, w_st, w_ed))
                elif op == "R":
                    p = np.random.randint(1, actual_len - 1)
                    w_st = max(p - half_window, 1)
                    w_ed = min(p + half_window, actual_len - 2)
                    input_ids_tmp[j, :-1] = input_ids[j, :]
                    attention_mask_tmp[j, :actual_len] = 1
                    op_info.append((op, p, w_st, w_ed))
                else:
                    assert 0

            attention_mask_tmp = attention_mask_tmp.to(self._device).unsqueeze(2)
            extended_attention_mask = self._clf_metric._model.get_extended_attention_mask(
                attention_mask_tmp, input_ids_tmp.size(), self._device)
            input_ids_tmp = input_ids_tmp.to(self._device)

            embs = self._clf_metric._model.bert.embeddings.word_embeddings(input_ids_tmp)
            embs.retain_grad()
            emb_input = self._clf_metric._model.bert.embeddings(inputs_embeds=embs)

            seed_rep_tmp = self._clf_metric._model.bert.encoder(
                emb_input, attention_mask=extended_attention_mask)[0]
            seed_rep = self._clf_metric._model.bert.pooler(seed_rep_tmp)
            err = seed_rep - source_rep
            err = (err * err).sum(dim=1)
            err.sum().backward()

            def find_toks(grad, j, st, ed):
                logits = torch.einsum(
                    "lk,lvk->lv", -embs.grad[j, st:ed],
                    (self._clf_metric._model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
                     - embs[j, st:ed].unsqueeze(1)))
                words = torch.multinomial(torch.softmax(logits, dim=1),
                                          1, replacement=True).squeeze(1)
                return list(words.detach().cpu().numpy())
                # return list(logits.argmax(dim=1).detach().cpu().numpy())

            candidate_sents = []

            for j in range(batch_size):
                op, p, w_st, w_ed = op_info[j]
                if op == "I":
                    toks = (list(input_ids[j, 1:w_st])
                            + find_toks(embs.grad, j, w_st, w_ed + 1)
                            + list(input_ids[j, w_ed:actual_len_list[j] - 1]))
                elif op == "D":
                    toks = (list(input_ids[j, 1:w_st])
                            + find_toks(embs.grad, j, w_st, w_ed + 1)
                            + list(input_ids[j, w_ed + 2:actual_len_list[j] - 1]))
                elif op == "R":
                    toks = (list(input_ids[j, 1:w_st])
                            + find_toks(embs.grad, j, w_st, w_ed + 1)
                            + list(input_ids[j, w_ed + 1:actual_len_list[j] - 1]))
                else:
                    assert 0
                candidate_sents.append(tostring(self._tokenizer, toks))

            # print(candidate_sents[0])

            previous_sents, decision_fn_state = l2_decision_fn(
                tokenizer=self._tokenizer, data_record=data_record, field_name=field_name,
                origin=original_text, previous_sents=previous_sents,
                candidate_sents=candidate_sents,
                burnin_weight=0.5,
                decision_fn_state=decision_fn_state,
                stats=self._stats,
                clf_metric=self._clf_metric,
                source_rep=source_rep)

            if (ii + 1) % 10 == 0:
                # ranks = np.argsort(-decision_fn_state)
                # keep_ids = ranks[:len(previous_sents) // 2]
                # skip_ids = ranks[len(previous_sents) // 2:]
                # for idx in skip_ids:
                #     tmp = np.random.choice(keep_ids)
                #     previous_sents[idx] = previous_sents[tmp]
                #     decision_fn_state[idx] = decision_fn_state[tmp]
                print(ii, np.min(-decision_fn_state))

        return previous_sents

    def paraphrase_example(self, data_record, field_name, n):
        self._stats = {
            "all": 0,
            "accept": 0
        }
        clipped_text = data_record[field_name]
        batch_size = self._strategy_config["batch_size"]

        sentences = []
        n_batches = math.ceil(n / batch_size)
        last_batch_size = n % batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size

        for idx in range(n_batches):
            burnin_steps = self._strategy_config["burnin_steps"]
            sampling_steps = self._strategy_config["sampling_steps"]

            batch = self._parallel_sequential_generation(
                clipped_text,
                data_record["seeds"],
                batch_size if idx != n_batches - 1 else last_batch_size,
                burnin_steps, sampling_steps, field_name, data_record)
            sentences += batch

        assert len(sentences) == n

        logger.info("Aggregated accept rate: %.2lf%%.",
                    self._stats["accept"] / self._stats["all"] * 100)
        return sentences[:n]
