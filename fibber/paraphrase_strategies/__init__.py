from fibber.paraphrase_strategies.bert_sampling_strategy import BertSamplingStrategy
from fibber.paraphrase_strategies.identity_strategy import IdentityStrategy
from fibber.paraphrase_strategies.random_strategy import RandomStrategy
from fibber.paraphrase_strategies.strategy_base import StrategyBase
from fibber.paraphrase_strategies.textattack_strategy import TextAttackStrategy

__all__ = ["IdentityStrategy", "RandomStrategy", "StrategyBase", "BertSamplingStrategy",
           "TextAttackStrategy"]
