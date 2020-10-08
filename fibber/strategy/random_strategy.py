import numpy as np

from .strategy_base import StrategyBase


class RandomStrategy(StrategyBase):
    """A baseline strategy. Randomly shuffle words in a sentence to generate paraphrases."""

    def paraphrase_example(self, data_record, field_name, n):
        text = data_record[field_name]
        tokens = text.split()
        ret = []
        for i in range(n):
            np.random.shuffle(tokens)
            ret.append(" ".join(tokens))

        return ret
