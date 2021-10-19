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

    def __init__(self, dataset_name, trainset, bert_ppl_gpu_id=-1, **kargs):
        """Initialize Bert perplexity model."""
        super(BertPerplexityMetric, self).__init__()

        if bert_ppl_gpu_id == -1:
            logger.warning("BertPerplexityMetric is running on CPU.")
            self._device = torch.device("cpu")
        else:
            logger.info("BertPerplexityMetric is running is running on GPU %d.", bert_ppl_gpu_id)
            self._device = torch.device("cuda:%d" % bert_ppl_gpu_id)

        logger.info("load bert perplexity model.")
        self._tokenizer, self._model = get_lm("ppl", dataset_name, trainset, self._device)
        self._model.to(self._device)

    def _get_ppl(self, sentences, data_record, paraphrase_field):
        """Compute the perplexity of sentences."""
        if paraphrase_field == "text0":
            batch_input = self._tokenizer(text=sentences, padding=True, max_length=200,
                                          truncation=True)
        else:
            assert paraphrase_field == "text1"
            batch_input = self._tokenizer(text=[data_record["text0"]] * len(sentences),
                                          text_pair=sentences,
                                          padding=True, max_length=200,
                                          truncation=True)

        with torch.no_grad():
            input_ids = torch.tensor(batch_input["input_ids"]).to(self._device)
            attention_mask = torch.tensor(batch_input["attention_mask"]).to(self._device)
            logits = self._model(
                input_ids=input_ids,
                token_type_ids=torch.tensor(batch_input["token_type_ids"]).to(
                    self._device),
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
            res = self._get_ppl([origin] + paraphrase_list, data_record, paraphrase_field)
            res = res[1:] / res[0]
        else:
            res = self._get_ppl(paraphrase_list, data_record, paraphrase_field)
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
