import numpy as np

from .strategy_base import StrategyBase


class RandomStrategy(StrategyBase):
    """docstring for RandomStrategy."""

    def paraphrase_example(self, data_record, field_name, n):
        text = data_record[field_name]
        tokens = text.split()
        ret = []
        for i in range(n):
            np.random.shuffle(tokens)
            ret.append(" ".join(tokens))

        return ret
