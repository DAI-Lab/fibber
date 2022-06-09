
from fibber.paraphrase_strategies.strategy_base import StrategyBase


class RemoveStrategy(StrategyBase):
    __abbr__ = "rm"

    def fit(self, trainset):
        self._clf = self._metric_bundle.get_target_classifier()
        self._tokenizer = self._metric_bundle.get_target_classifier()._tokenizer

    def paraphrase_example(self, data_record, n):
        text = data_record[self._field]
        tokens = self._tokenizer.tokenize(text)

        for i in range(len(tokens)):
            current = self._tokenizer.convert_tokens_to_string(tokens[:i] + tokens[i + 1:])
            pred = self._clf.predict_example(None, current)
            if pred != data_record["label"]:
                return [current], i + 1

        return [text], len(tokens)
