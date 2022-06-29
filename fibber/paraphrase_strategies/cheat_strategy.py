from fibber.paraphrase_strategies.strategy_base import StrategyBase


class CheatStrategy(StrategyBase):
    """A baseline paraphrase strategy. Just return the reference."""

    __abbr__ = "cheat"

    def paraphrase_example(self, data_record, n):
        return [data_record["ref"]]
