import numpy as np

from fibber.paraphrase_strategies.strategy_base import StrategyBase


class RandomStrategy(StrategyBase):
    """Randomly shuffle words in a sentence to generate paraphrases."""

    __abbr__ = "rand"

    def paraphrase_example(self, data_record, field_name, n):
        text = data_record[field_name]
        tokens = text.split()
        ret = []
        for i in range(n):
            np.random.shuffle(tokens)
            ret.append(" ".join(tokens))

        return ret
