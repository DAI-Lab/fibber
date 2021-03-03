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
        ("enforce_similarity", str, "1", "enforce the sampling similarity.")
    ]

    def fit(self, trainset):
        # load BERT language model.

        self._use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")
        self._tokenizer, lm = get_lm(
            "nartune", self._output_dir, trainset, self._device,
            self._strategy_config["lm_steps"], use_metric=self._use_metric)
        if isinstance(lm, list):
            self._bert_lms = lm
        else:
            self._bert_lm = lm
            self._bert_lm.to(self._device)

        self._text_parser = TextParser()

    def enforce_similarity(self, origin, batch_tensor, batch_tensor_new):
        use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")
        prev_sim = use_metric.measure_batch(
            origin, [self._tokenizer.decode(x[1:-1]) for x in batch_tensor])
        curr_sim = use_metric.measure_batch(
            origin, [self._tokenizer.decode(x[1:-1]) for x in batch_tensor_new])

        return torch.logical_or((torch.tensor(curr_sim) > torch.tensor(prev_sim)),
                                (torch.tensor(curr_sim) > 0.85)).int().to(self._device)

    def paraphrase_example(self, data_record, field_name, n):
        seeds = self._text_parser.phrase_level_shuffle(data_record["text0"], n)

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

        for iter_id, replace_percentage in enumerate([0.3, 0.3, 0.3, 0.2, 0.1, 0.05]):
            num_pos = max(int(replace_percentage * (tensor_len - 2)), 1)

            pos = F.one_hot(
                torch.topk(previous_logits, k=num_pos, dim=1, largest=False)[1],
                num_classes=tensor_len).sum(dim=1)

            with torch.no_grad():
                batch_tensor_masked = (batch_tensor
                                       * (1 - pos) + pos * self._tokenizer.mask_token_id)
                logits = self._bert_lm(sentence_embeds=ori_emb, input_ids=batch_tensor_masked)[0]
                batch_tensor_new = logits.argmax(dim=2)

            if self._strategy_config["enforce_similarity"] == "1" and iter_id != 0:
                update = self.enforce_similarity(
                    data_record["text0"], batch_tensor, batch_tensor_new)
            else:
                update = torch.ones(n).long().to(self._device)

            previous_logits_new = (
                logits * F.one_hot(
                    batch_tensor, num_classes=self._tokenizer.vocab_size)).sum(dim=-1)
            previous_logits_new[:, 1] = 1e8
            previous_logits_new[:, -1] = 1e8

            batch_tensor = (update[:, None] * batch_tensor_new
                            + (1 - update[:, None]) * batch_tensor)
            previous_logits = (update[:, None] * previous_logits
                               + (1 - update[:, None]) * previous_logits_new)
        batch_tensor = batch_tensor.detach().cpu()[:, 1:-1]
        return [self._tokenizer.decode(x) for x in batch_tensor]
