"""This metric computes the perplexity ratio ppl(paraphrase) / ppl(original text).

The perplexity is estimated using GPT2 model. This metric can reveal the meaningfulness of a
sentence.
"""
import numpy as np
import torch

from fibber import log
from fibber.metrics.bert_lm_utils import get_lm
from fibber.metrics.metric_base import MetricBase
from expiringdict import ExpiringDict

logger = log.setup_custom_logger(__name__)


class BertPerplexityMetric(MetricBase):
    """This metric computes the perplexity of paraphrased text divided by the perplexity of
    original text. The perplexity is measured using BERT model.
    """

    def __init__(self, dataset_name, trainset, bert_ppl_gpu_id=-1, bert_ppl_filter=-1, **kargs):
        """Initialize Bert perplexity model."""
        super(BertPerplexityMetric, self).__init__()

        if bert_ppl_gpu_id == -1:
            logger.warning("BertPerplexityMetric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("BertPerplexityMetric is running is running on GPU %d.", bert_ppl_gpu_id)
            self._device = torch.device("cuda:%d" % bert_ppl_gpu_id)

        logger.info("load bert perplexity model.")
        self._tokenizer, self._model = get_lm("ppl", dataset_name, trainset, self._device,
                                              filter=bert_ppl_filter)
        self._model.to(self._device)
        self._data_filter = bert_ppl_filter
        self._name_suffix = ""
        if self._data_filter != -1:
            self._name_suffix = "-exclude-" + trainset["label_mapping"][self._data_filter]
        self.cache = ExpiringDict(max_len=100000, max_age_seconds=1800)

    @property
    def lm_model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def __repr__(self):
        return self.__class__.__name__ + self._name_suffix

    def _get_ppl(self, sentences):
        """Compute the perplexity of sentences."""
        batch_input = self._tokenizer(text=sentences, padding=True)

        with torch.no_grad():
            input_ids = torch.tensor(batch_input["input_ids"]).to(self._device)
            attention_mask = torch.tensor(batch_input["attention_mask"]).to(self._device)
            token_type_ids = torch.tensor(batch_input["token_type_ids"]).to(self._device)
            logits = self._model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )[0]
            logpw = torch.gather(torch.log_softmax(logits[:, :-1], dim=-1), dim=2,
                                 index=input_ids[:, 1:].unsqueeze(2)).squeeze(2)

            ppl = torch.exp(-(logpw * attention_mask[:, 1:]).sum(dim=1)
                            / (attention_mask.sum(dim=1) - 1))
            ppl = ppl.detach().cpu().numpy()

        return ppl

    def measure_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0",
                      use_ratio=False):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.
            use_ratio (bool): whether to return ppl ratio.
        Returns:
            (list): a list containing the USE similarity metric for each paraphrase.
        """
        if use_ratio:
            res = self._get_ppl([origin] + paraphrase_list)
            res = res[1:] / res[0]
        else:
            res = self._get_ppl(paraphrase_list)
        return [float(x) for x in res]

    def measure_multiple_examples(self, origin_list, paraphrase_list,
                                  data_record_list=None, paraphrase_field="text0",
                                  use_ratio=False):
        assert len(origin_list) == len(paraphrase_list)
        if use_ratio:
            ppls = self._get_ppl(origin_list + paraphrase_list)
            res = ppls[len(origin_list):] / ppls[:len(origin_list)]
        else:
            res = self._get_ppl(paraphrase_list)
        return [float(x) for x in res]

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0",
                        use_ratio=False):
        """Compute the perplexity ratio.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.
            use_ratio (bool): whether to return ppl ratio.
        """
        return self.measure_batch(origin, [paraphrase], data_record, paraphrase_field,
                                  use_ratio=use_ratio)[0]

    # def perplexity_filter(self, sentences, bar=-3):
    #     batch_input = self._tokenizer(text=sentences, padding=True, return_tensors="np")
    #
    #     with torch.no_grad():
    #         input_ids = torch.tensor(batch_input["input_ids"]).to(self._device)
    #         attention_mask = torch.tensor(batch_input["attention_mask"]).to(self._device)
    #         token_type_ids = torch.tensor(batch_input["token_type_ids"]).to(self._device)
    #         logits = self._model(
    #             input_ids=input_ids,
    #             token_type_ids=token_type_ids,
    #             attention_mask=attention_mask
    #         )[0]
    #         logpw = torch.gather(torch.log_softmax(logits[:, :-1], dim=-1), dim=2,
    #                              index=input_ids[:, 1:].unsqueeze(2)).squeeze(2).detach().cpu().numpy()
    #
    #         filtered_sent = []
    #         cc = 0
    #         for sent_id in range(len(sentences)):
    #             sent_len = batch_input["attention_mask"][sent_id].sum() - 1
    #             avg = logpw[sent_id, :sent_len].mean()
    #             sent = batch_input["input_ids"][sent_id, 1:sent_len+1]
    #             for p in range(sent_len):
    #                 if logpw[sent_id, p] - avg < bar:
    #                     sent[p] = self._tokenizer.mask_token_id
    #                     cc += 1
    #             sent = [item for item in sent if item != self._tokenizer.mask_token_id]
    #             filtered_sent.append(self._tokenizer.convert_tokens_to_string(
    #                 self._tokenizer.convert_ids_to_tokens(sent[:-1])))
    #         print("avg remove", cc / len(sentences))
    #     return filtered_sent

    # def _filter(self, sentence, bar):
    #     tokens = sentence.split()
    #     sentence_set = [sentence]
    #     for idx in range(len(tokens)):
    #         sentence_set.append(" ".join(tokens[:idx] + tokens[idx+1:]))
    #
    #     ppls = []
    #     st = 0
    #     while st < len(sentence_set):
    #         ed = st + 64
    #         ppls += list(self._get_ppl(sentence_set[st:ed]))
    #         st = ed
    #
    #     ret = []
    #     for i in range(len(tokens)):
    #         if ppls[i + 1] - ppls[0] > bar:
    #             ret.append(tokens[i])
    #     return " ".join(ret)

    def _filter(self, sentence, bar):
        tokens = sentence.split()
        sentence_set = []
        for idx in range(len(tokens)):
            st = max(0, idx - 5)
            ed = min(idx + 6, len(tokens))
            s1 = " ".join(tokens[st:ed])
            s2 = " ".join(tokens[st:idx] + tokens[idx + 1:ed])
            if s1 not in self.cache:
                sentence_set.append(s1)
            if s2 not in self.cache:
                sentence_set.append(s2)
        ppls = []
        st = 0
        while st < len(sentence_set):
            ed = st + 64
            ppls += list(self._get_ppl(sentence_set[st:ed]))
            st = ed

        for s, ppl in zip(sentence_set, ppls):
            self.cache[s] = ppl

        ret = []
        for i in range(len(tokens)):
            st = max(0, idx - 5)
            ed = min(idx + 6, len(tokens))
            ppl = self.cache[" ".join(tokens[st:ed])]
            ppl_rm = self.cache[" ".join(tokens[st:idx] + tokens[idx + 1:ed])]

            if ppl_rm - ppl > bar:
                ret.append(tokens[i])
        return " ".join(ret)

    def perplexity_filter(self, sentences, bar=0):
        result = []
        for sentence in sentences:
            result.append(self._filter(sentence, bar))

        return result
