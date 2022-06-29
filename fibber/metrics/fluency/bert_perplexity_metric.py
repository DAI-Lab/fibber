"""This metric computes the perplexity ratio ppl(paraphrase) / ppl(original text).

The perplexity is estimated using GPT2 model. This metric can reveal the meaningfulness of a
sentence.
"""
import torch

from fibber import log
from fibber.metrics.bert_lm_utils import get_lm
from fibber.metrics.metric_base import MetricBase

logger = log.setup_custom_logger(__name__)


class BertPerplexityMetric(MetricBase):
    """This metric computes the perplexity of paraphrased text divided by the perplexity of
    original text. The perplexity is measured using BERT model.
    """

    def __init__(self, dataset_name, trainset, bert_ppl_gpu_id=-1, bert_ppl_filter=-1, **kwargs):
        """Initialize Bert perplexity model."""
        super(BertPerplexityMetric, self).__init__(**kwargs)

        if bert_ppl_gpu_id == -1:
            logger.warning("BertPerplexityMetric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("BertPerplexityMetric is running is running on GPU %d.", bert_ppl_gpu_id)
            self._device = torch.device("cuda:%d" % bert_ppl_gpu_id)

        logger.info("load bert perplexity model.")
        self._tokenizer, self._model = get_lm("ppl", dataset_name, trainset, self._device,
                                              filter=bert_ppl_filter, select_field=self._field)
        self._model.to(self._device)
        self._data_filter = bert_ppl_filter
        self._name_suffix = ""
        if self._data_filter != -1:
            self._name_suffix = "-exclude-" + trainset["label_mapping"][self._data_filter]

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

    def _measure_batch(self, origin, paraphrase_list, data_record=None, field="text0",
                       use_ratio=False, **kwargs):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            field (str): the field name to paraphrase.
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

    def _measure_multiple_examples(self, origin_list, paraphrase_list,
                                   data_record_list=None, field="text0",
                                   use_ratio=False, **kwargs):
        assert len(origin_list) == len(paraphrase_list)
        if use_ratio:
            ppls = self._get_ppl(origin_list + paraphrase_list)
            res = ppls[len(origin_list):] / ppls[:len(origin_list)]
        else:
            res = self._get_ppl(paraphrase_list)
        return [float(x) for x in res]

    def _measure_example(self, origin, paraphrase, data_record=None, use_ratio=False, **kwargs):
        """Compute the perplexity ratio.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            use_ratio (bool): whether to return ppl ratio.
        """
        return self.measure_batch(origin, [paraphrase], data_record, use_ratio=use_ratio)[0]
