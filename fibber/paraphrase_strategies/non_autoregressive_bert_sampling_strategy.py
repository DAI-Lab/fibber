import numpy as np
import torch
from torch.nn import functional as F

from fibber import log
from fibber.paraphrase_strategies.bert_sampling_utils_lm import get_lm
from fibber.paraphrase_strategies.bert_sampling_utils_text_parser import TextParser
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


def compute_emb(lm_model, seq, tok_type, sent_emb):
    tmp_emb = lm_model.use_linear(sent_emb)  # batch * 768
    sent = lm_model.bert.embeddings.word_embeddings(seq)
    inp_emb = torch.cat([tmp_emb[:, None, :], sent[:, 1:]], dim=1)

    return lm_model.bert.embeddings(None, tok_type, inputs_embeds=inp_emb)


class NonAutoregressiveBertSamplingStrategy(StrategyBase):
    __abbr__ = "nabs"
    __hyperparameters__ = [
        ("lm_steps", int, 5000, "lm training steps."),
        ("enforce_similarity", str, "1", "enforce the sampling similarity."),
        ("num_decode_iter", int, 10, "number of iterations to decode the sentence."),
        ("clf_weight", float, 1, ""),
        ("use_weight", float, 10, ""),
        ("no_enforce_iter", int, 3, ""),
        ("max_mask_rate", float, 0.5, ""),
    ]

    def fit(self, trainset):
        # load BERT language model.

        self._use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")
        self._tokenizer, lm = get_lm(
            "nartune", self._output_dir, trainset, self._device,
            self._strategy_config["lm_steps"], use_metric=self._use_metric, lm_bs=16)
        if isinstance(lm, list):
            self._bert_lms = lm
        else:
            self._bert_lm = lm
            self._bert_lm.to(self._device)

        self._text_parser = TextParser()

    def enforce_similarity(self, origin, batch_tensor, batch_tensor_new, data_record):
        use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")
        prev_sim = use_metric.measure_batch(
            origin, [self._tokenizer.decode(x[1:-1]) for x in batch_tensor])
        curr_sim = use_metric.measure_batch(
            origin, [self._tokenizer.decode(x[1:-1]) for x in batch_tensor_new])
        prev_sim = np.asarray(prev_sim)
        curr_sim = np.asarray(curr_sim)
        alpha = -self._strategy_config("use_weight") * (
            np.asarray(curr_sim < prev_sim, dtype="float32")
            * np.max(0.90 - curr_sim, 0) ** 2)

        bert_metric = self._metric_bundle.get_target_classifier()
        curr_dist = bert_metric.predict_dist_batch(
            origin, [self._tokenizer.decode(x[1:-1]) for x in batch_tensor_new])
        prev_dist = bert_metric.predict_dist_batch(
            origin, [self._tokenizer.decode(x[1:-1]) for x in batch_tensor])

        label = data_record["label"]
        curr_dist_correct = (curr_dist[:, label]).copy()
        curr_dist[:, label] = -1e8
        curr_dist_incorrect = np.max(curr_dist, axis=1)
        curr_dist_diff = np.maximum(np.exp(curr_dist_correct) - np.exp(curr_dist_incorrect), 0)

        prev_dist_correct = (prev_dist[:, label]).copy()
        prev_dist[:, label] = -1e8
        prev_dist_incorrect = np.max(prev_dist, axis=1)
        prev_dist_diff = np.maximum(np.exp(prev_dist_correct) - np.exp(prev_dist_incorrect), 0)

        alpha += -self._strategy_config("clf_weight") * np.max(curr_dist_diff - prev_dist_diff, 0)

        return torch.tensor(np.random.rand(alpha.size) < np.exp(alpha)).int().to(self._device)

    def paraphrase_example(self, data_record, field_name, n):
        # seeds = self._text_parser.phrase_level_shuffle(data_record["text0"], n)
        seeds = [data_record["text0"] for i in range(n)]
        seq = [self._tokenizer.tokenize(x) for x in seeds]
        seq_len_max = max([len(x) for x in seq])

        # TODO add support to NLI datasets.
        seq = [["[CLS]"] + x + ["[MASK]"] * (seq_len_max - len(x)) + ["[SEP]"] for x in seq]
        tensor_len = seq_len_max + 2

        batch_tensor = torch.tensor(
            [self._tokenizer.convert_tokens_to_ids(x) for x in seq]).to(self._device)

        previous_logits = torch.rand_like(batch_tensor.float())
        previous_logits[:, 0] = 1e8
        previous_logits[:, -1] = 1e8

        ori_emb = self._use_metric.model([data_record["text0"]]).numpy()
        ori_emb = np.tile(ori_emb, (n, 1))

        update_list = []

        num_decode_iter = self._strategy_config["num_decode_iter"]
        for iter_id in range(num_decode_iter):
            num_pos = max(
                int(self._strategy_config["max_mask_rate"] * (tensor_len - 2)
                    * (num_decode_iter - iter_id) / num_decode_iter), 1)

            pos = F.one_hot(
                torch.topk(previous_logits, k=num_pos, dim=1, largest=False)[1],
                num_classes=tensor_len).sum(dim=1)

            with torch.no_grad():
                batch_tensor_masked = (batch_tensor
                                       * (1 - pos) + pos * self._tokenizer.mask_token_id)
                logits = self._bert_lm(sentence_embeds=ori_emb, input_ids=batch_tensor_masked)[0]
                batch_tensor_new = (logits.argmax(dim=2) * pos + batch_tensor * (1 - pos))
                previous_logits_new = (F.log_softmax(logits, dim=2).max(dim=2)[0]
                                       * pos + previous_logits * (1 - pos))

            if (self._strategy_config["enforce_similarity"] == "1"
                    and iter_id >= self._strategy_config["no_enforce_iter"]):
                update = self.enforce_similarity(
                    data_record["text0"], batch_tensor, batch_tensor_new, data_record)
                update_list.append(update.float().mean().detach().cpu().numpy())
            else:
                update = torch.ones(n).long().to(self._device)

            batch_tensor = (update[:, None] * batch_tensor_new
                            + (1 - update[:, None]) * batch_tensor)
            previous_logits = (update[:, None] * previous_logits
                               + (1 - update[:, None]) * previous_logits_new)
        batch_tensor = batch_tensor.detach().cpu()[:, 1:-1]

        print(np.mean(update_list))
        return [self._tokenizer.decode(x) for x in batch_tensor]
