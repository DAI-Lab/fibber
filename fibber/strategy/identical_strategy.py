from .strategy_base import StrategyBase


class IdenticalStrategy(StrategyBase):
    """A baseline strategy. Just return the original sentence."""

    def paraphrase_example(self, data_record, field_name, n):
        return [data_record[field_name]]
