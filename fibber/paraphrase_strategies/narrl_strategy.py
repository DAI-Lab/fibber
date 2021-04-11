import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from nltk import sent_tokenize

from fibber import log
from fibber.datasets.dataset_utils import KeywordsExtractor
from fibber.paraphrase_strategies.bert_sampling_utils_lm import get_lm
from fibber.paraphrase_strategies.strategy_base import StrategyBase

logger = log.setup_custom_logger(__name__)


class NARRLStrategy(StrategyBase):
    """The non-autoregressive paraphrasing strategy using reinforcement learning. """

    __abbr__ = "nr"
    __hyperparameters__ = [
        ("lm_steps", int, 5000, "lm training steps."),
        ("lm_bs", int, 16, "lm batch size."),
        ("num_decode_iter", int, 10, "number of decoding iterations."),
        ("max_mask_rate", float, 1, "maximum ratio of masking."),
        ("num_keywords", int, 5, "number of keywords"),
        ("split_sentences", str, "0", "0 for paragraph level model.")
    ]

    def fit(self, trainset):
        # load BERT language model.
        logger.info("Load non-autoregressive langauge model for NARRL.")

        use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")

        self._tokenizer, self._bert_lm = get_lm(
            "narrl", self._output_dir, trainset, self._device,
            self._strategy_config["lm_steps"], lm_bs=self._strategy_config["lm_bs"],
            rl_bs=1,
            use_metric=use_metric, num_keywords=self._strategy_config["num_keywords"],
            split_sentences=(self._strategy_config["split_sentences"] == "1"),
            max_mask_rate=self._strategy_config["max_mask_rate"])

        self._bert_lm.to(self._device)
        self._keyword_extractor = KeywordsExtractor()
        self._use_metric = self._metric_bundle.get_metric("USESemanticSimilarityMetric")

    def _paraphrase_example(self, data_record, field_name, n):
        keywords = self._keyword_extractor.extract_keywords(
            data_record["text0"], n=self._strategy_config["num_keywords"])
        print(keywords)
        ori_len = len(self._tokenizer.tokenize(data_record["text0"]))

        s0s = []
        s1s = []
        for i in range(n):
            if len(keywords) > 0:
                random.shuffle(keywords)
            s0s.append("[PAD] " + ",".join(keywords))
            gen_len = max(ori_len + np.random.randint(-2, 3), 1)
            s1s.append("[MASK] " * gen_len)

        batch_input = self._tokenizer(s0s, s1s, padding=True)

        input_ids = torch.tensor(batch_input["input_ids"]).to(self._device)
        token_type_ids = torch.tensor(batch_input["token_type_ids"]).to(self._device)
        attention_mask = torch.tensor(batch_input["attention_mask"]).to(self._device)
        modifiable_part = token_type_ids * attention_mask * (
            input_ids != self._tokenizer.sep_token_id).long()

        tensor_len = input_ids.size(1)

        previous_logits = torch.rand_like(input_ids.float())
        previous_logits = previous_logits * modifiable_part + 1e8 * (1 - modifiable_part)

        ori_emb = self._use_metric.model([data_record["text0"]]).numpy()
        ori_emb = np.tile(ori_emb, (n, 1))

        num_decode_iter = self._strategy_config["num_decode_iter"]
        for iter_id in range(num_decode_iter):
            num_pos = max(
                int(self._strategy_config["max_mask_rate"] * (ori_len + 2)
                    * (num_decode_iter - iter_id) / num_decode_iter), 1)

            pos = F.one_hot(
                torch.topk(previous_logits, k=num_pos, dim=1, largest=False)[1],
                num_classes=tensor_len).sum(dim=1) * modifiable_part

            with torch.no_grad():
                input_ids_masked = (input_ids * (1 - pos) + pos * self._tokenizer.mask_token_id)
                logits = self._bert_lm(sentence_embeds=ori_emb,
                                       input_ids=input_ids_masked,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]
                input_ids_new = (logits.argmax(dim=2) * pos + input_ids * (1 - pos))
                previous_logits_new = (F.log_softmax(logits, dim=2).max(dim=2)[0]
                                       * pos + previous_logits * (1 - pos))

            update = torch.ones(n).long().to(self._device)

            input_ids = (update[:, None] * input_ids_new
                         + (1 - update[:, None]) * input_ids)
            previous_logits = (update[:, None] * previous_logits
                               + (1 - update[:, None]) * previous_logits_new)

        input_ids = input_ids.detach().cpu().numpy()
        modifiable_part = modifiable_part.detach().cpu().numpy()

        ret = []
        for i in range(n):
            t = []
            for wid, mm in zip(input_ids[i], modifiable_part[i]):
                if mm == 1:
                    t.append(wid)
            ret.append(self._tokenizer.decode(t))

        return ret

    def paraphrase_example(self, data_record, field_name, n):
        if self._strategy_config["split_sentences"] == "1":
            sents = sent_tokenize(data_record["text0"])
        else:
            sents = [data_record["text0"]]

        result = [""] * n
        for sent in sents:
            data_record_tmp = copy.copy(data_record)
            data_record_tmp["text0"] = sent
            result_tmp = self._paraphrase_example(data_record_tmp, field_name, n)
            for i in range(n):
                result[i] += result_tmp[i]

        return result
