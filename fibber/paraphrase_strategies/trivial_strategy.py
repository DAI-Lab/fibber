import json
import os.path

import numpy as np

from fibber.paraphrase_strategies.strategy_base import StrategyBase


def construct_candidates(sent, target_word):
    ret = [sent]
    toks = sent.split()
    for i in range(len(toks)):
        ret.append(" ".join(toks[:i] + [target_word] + toks[i+1:]))
    return ret


class TrivialStrategy(StrategyBase):
    """A baseline paraphrase strategy. Just return the original sentence."""

    __abbr__ = "tr"
    __hyperparameters__ = [
        ("filename", str, None, "the file of word adversarial power.")
    ]

    def fit(self, trainset):
        self._sim_metric = self._metric_bundle.get_metric("USESimilarityMetric")
        self._clf_metric = self._metric_bundle.get_target_classifier()
        self._ppl_metric = self._metric_bundle.get_metric("BertPerplexityMetric")

        m = len(trainset["label_mapping"])

        filename = self._strategy_config["filename"]
        if "ag-" in filename:
            filename = filename.replace("ag-", "ag_no_title-")

        if not os.path.exists(filename):
            if filename.endswith("-bert.json"):
                filename = filename.replace("-bert.json", ".json")

        with open(filename) as f:
            trivial_words = json.load(f)
            if "trivial" in trivial_words:
                trivial_words = trivial_words["trivial"]
            else:
                trivial_words = trivial_words["sac"]

        self._trivial_candidates = trivial_words[:50]
        print(self._trivial_candidates)

    def paraphrase_example(self, data_record, field_name, n):
        if self._clf_metric.predict_example(
                data_record["text0"], data_record["text0"]) != data_record["label"]:
            return [data_record[field_name]], 0

        mis_clf_sents = []
        all_sents = []
        for item in self._trivial_candidates:
            target_word = item[0]
            all_sents += construct_candidates(data_record["text0"], target_word)

        np.random.shuffle(all_sents)

        for st in range(0, len(all_sents), 64):
            sents = all_sents[st:st+64]
            predicts = self._clf_metric.predict_batch(data_record["text0"], sents)
            for pp, ss in zip(predicts, sents):
                if pp != data_record["label"]:
                    mis_clf_sents.append(ss)
            if len(mis_clf_sents) >= 50:
                break

        if len(mis_clf_sents) > 0:
            ppls = self._ppl_metric.measure_batch(data_record["text0"], mis_clf_sents,
                                                  use_ratio=True)
            sims = self._sim_metric.measure_batch(data_record["text0"], mis_clf_sents)

            score = 3 * np.asarray(ppls) + 20 * (1 - np.asarray(sims))
            idx = np.argmin(score)
            return [mis_clf_sents[idx]], 0
        else:
            return [data_record[field_name]], 0

