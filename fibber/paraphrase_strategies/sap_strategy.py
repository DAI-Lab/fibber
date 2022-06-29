
import numpy as np

from fibber.datasets import subsample_dataset
from fibber.metrics.classifier.transformer_classifier import TransformerClassifier
from fibber.paraphrase_strategies.sap_utils_euba import solve_euba
from fibber.paraphrase_strategies.strategy_base import StrategyBase


def construct_candidates(sent, target_word):
    ret = [sent]
    toks = sent.split()
    for i in range(len(toks)):
        ret.append(" ".join(toks[:i] + [target_word] + toks[i + 1:]))
    return ret


class SapStrategy(StrategyBase):
    """A baseline paraphrase strategy. Just return the original sentence."""

    __abbr__ = "sap"
    __hyperparameters__ = [
        ("euba_subsample_size", int, 1000, "the number of training examples to compute kappa.")
    ]

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle):
        super(SapStrategy, self).__init__(arg_dict, dataset_name, strategy_gpu_id,
                                          output_dir, metric_bundle)
        self._sim_metric = None
        self._clf_metric = None
        self._ppl_metric = None
        self._adversarial_word_candidates = None

    def fit(self, trainset):
        self._sim_metric = self._metric_bundle.get_metric("USESimilarityMetric")
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._ppl_metric = self._metric_bundle.get_metric("BertPerplexityMetric")

        if not isinstance(self._clf_metric, TransformerClassifier):
            raise RuntimeError("Sap attack only supports TransformerClassifier.")

        model, tokenizer = self._clf_metric.get_model_and_tokenizer()
        vocabulary = list(tokenizer.vocab.items())
        model_init = self._clf_metric.get_model_init()

        if model_init.startswith("bert-") or model_init.startswith("distilbert-"):
            vocabulary = sorted([item for item in vocabulary if item[0].encode("utf8").isalpha()],
                                key=lambda x: x[1])
        elif model_init.startswith("roberta-"):
            vocabulary = sorted([item for item in vocabulary
                                 if item[0].encode("utf8").isalpha() or (
                                     item[0][0] == "Ġ" and item[0][1:].encode("utf8").isalpha())],
                                key=lambda x: x[1])
        else:
            raise RuntimeError("Only support BERT, distilBERT, RoBERTa.")

        kappa, _ = solve_euba(
            clf_model=model, tokenizer=tokenizer, vocabulary=vocabulary,
            val_set=subsample_dataset(trainset, self._strategy_config["euba_subsample_size"]),
            use_mask=True, use_top1=False, early_stop=512, batch_size=32,
            device=self._clf_metric.get_device())

        sac_result = [(w[0], float(eps))
                      for w, eps in zip(vocabulary, kappa)]
        sac_result = sorted(sac_result, key=lambda x: (-x[1], x[0]))

        self._adversarial_word_candidates = sac_result[:50]

    def paraphrase_example(self, data_record, n):
        if self._clf_metric.predict_example(
                data_record["text0"], data_record["text0"]) != data_record["label"]:
            return [data_record[self._field]], 0

        mis_clf_sents = []
        all_sents = []
        for item in self._adversarial_word_candidates:
            target_word = item[0]
            all_sents += construct_candidates(data_record["text0"], target_word)

        np.random.shuffle(all_sents)

        counter = 0
        for st in range(0, len(all_sents), 64):
            sents = all_sents[st:st + 64]
            predicts = self._clf_metric.predict_batch(data_record["text0"], sents)
            counter += len(sents)
            for pp, ss in zip(predicts, sents):
                if pp != data_record["label"]:
                    mis_clf_sents.append(ss)
            if len(mis_clf_sents) >= 50:
                break

        if len(mis_clf_sents) > 0:
            ppls = self._ppl_metric.measure_batch(
                data_record["text0"], mis_clf_sents, use_ratio=True)
            sims = self._sim_metric.measure_batch(data_record["text0"], mis_clf_sents)

            score = 3 * np.asarray(ppls) + 20 * (1 - np.asarray(sims))
            idx = np.argmin(score)
            return [mis_clf_sents[idx]], counter
        else:
            return [data_record[self._field]], counter
