import os

import torch

from fibber import get_root_dir, log

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)


class DefenseStrategyBase(object):
    """Base class for Tuning strategy"""

    __abbr__ = "defense_base"
    __hyperparameters__ = []

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, defense_desc,
                 metric_bundle, attack_strategy, field):
        """Initialize the paraphrase_strategies.

        This function initialize the ``self._strategy_config``, ``self._metric_bundle``,
        ``self._device``, ``self._defense_desc``, ``self._dataset_name``.

        **You should not overwrite this function.**

        * self._strategy_config (dict): a dictionary that stores the strategy name and all
          hyperparameter values. The dict is also saved to the results.
        * self._metric_bundle (MetricBundle): the metrics that will be used to evaluate
          paraphrases. Strategies can compute metrics during paraphrasing.
        * self._device (torch.Device): any computation that requires a GPU accelerator should
          use this device.
        * self._defense_desc (str): the dir name where the defense will save files.
        * self._dataset_name (str): the dataset name.

        Args:
            arg_dict (dict): all args load from command line.
            dataset_name (str): the name of the dataset.
            strategy_gpu_id (int): the gpu id to run the strategy.
            metric_bundle (MetricBundle): a MetricBundle object.
            attack_strategy (ParaphraseStrategyBase or None): the attack strategy. Used in some
                defense methods.
            field (str): the field that perturbation can happen.
        """
        super(DefenseStrategyBase, self).__init__()

        # paraphrase_strategies config will be saved to the results.
        self._strategy_config = dict()

        for p_name, p_type, p_default, p_help in self.__hyperparameters__:
            arg_name = "%s_%s" % (self.__abbr__, p_name)
            if arg_name not in arg_dict:
                logger.warning("%s_%s not found in args.", self.__abbr__, p_name)
                p_value = p_default
            else:
                p_value = arg_dict[arg_name]

            assert p_name not in self._strategy_config
            self._strategy_config[p_name] = p_value

        self._strategy_config["strategy_name"] = str(self)

        self._metric_bundle = metric_bundle
        self._attack_strategy = attack_strategy

        if strategy_gpu_id == -1:
            logger.warning("%s is running on CPU." % str(self))
            self._device = torch.device("cpu")
        else:
            logger.info("%s is running on GPU %d.", str(self), strategy_gpu_id)
            self._device = torch.device("cuda:%d" % strategy_gpu_id)

        self._dataset_name = dataset_name
        self._defense_desc = defense_desc
        self._defense_save_path = os.path.join(get_root_dir(), self._defense_desc, dataset_name)
        os.makedirs(self._defense_save_path, exist_ok=True)
        self._classifier = self._metric_bundle.get_target_classifier()
        self._field = field

    @classmethod
    def add_parser_args(cls, parser):
        """create commandline args for all hyperparameters in ``__hyperparameters__``.

        Args:
            parser: an arg parser.
        """
        logger.info("%s has %d args to set from command line.",
                    cls.__name__, len(cls.__hyperparameters__))

        for p_name, p_type, p_default, p_help in cls.__hyperparameters__:
            parser.add_argument("--%s_%s" % (cls.__abbr__, p_name), type=p_type,
                                default=p_default, help=p_help)

    def __repr__(self):
        return self.__class__.__name__

    def fit(self, trainset):
        """Fit the paraphrase strategy on a training set.

        Args:
            trainset (dict): a fibber dataset.
        """
        return self._classifier

    def load(self, trainset):
        return self._classifier
