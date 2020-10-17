from fibber.paraphrase_strategies.strategy_base import StrategyBase


class IdentityStrategy(StrategyBase):
    """A baseline paraphrase strategy. Just return the original sentence."""

    def paraphrase_example(self, data_record, field_name, n):
        return [data_record[field_name]]
