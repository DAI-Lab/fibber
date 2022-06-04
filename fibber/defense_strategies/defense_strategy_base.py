from fibber import log
import torch
logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)


class DefenseStrategyBase(object):
    """Base class for Tuning strategy"""

    __abbr__ = "defense_base"
    __hyperparameters__ = []

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, metric_bundle):
        """Initialize the paraphrase_strategies.

        This function initialize the ``self._strategy_config``, ``self._metric_bundle``,
        ``self._device``, ``self._output_dir``, ``self._dataset_name``.

        **You should not overwrite this function.**

        * self._strategy_config (dict): a dictionary that stores the strategy name and all
          hyperparameter values. The dict is also saved to the results.
        * self._metric_bundle (MetricBundle): the metrics that will be used to evaluate
          paraphrases. Strategies can compute metrics during paraphrasing.
        * self._device (torch.Device): any computation that requires a GPU accelerator should
          use this device.
        * self._output_dir (str): the dir name where the strategy can save files.
        * self._dataset_name (str): the dataset name.

        Args:
            arg_dict (dict): all args load from command line.
            dataset_name (str): the name of the dataset.
            strategy_gpu_id (int): the gpu id to run the strategy.
            metric_bundle (MetricBundle): a MetricBundle object.
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

        if strategy_gpu_id == -1:
            logger.warning("%s is running on CPU." % str(self))
            self._device = torch.device("cpu")
        else:
            logger.info("%s is running on GPU %d.", str(self), strategy_gpu_id)
            self._device = torch.device("cuda:%d" % strategy_gpu_id)

        self._dataset_name = dataset_name

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

    def input_manipulation(self):
        pass

    def fine_tune_classifier(self,
                             metric_bundle,
                             paraphrase_strategy,
                             train_set,
                             num_paraphrases_per_text,
                             tuning_steps,
                             tuning_batch_size=32,
                             num_sentences_to_rewrite_per_step=20,
                             num_updates_per_step=5,
                             period_save=1000):
        """Fine tune the classifier using given data.

        Args:
            metric_bundle (MetricBundle): a metric bundle including the target classifier.
            paraphrase_strategy (StrategyBase): a paraphrase strategy to fine tune the classifier.
            train_set (dict): the training set of the classifier.
            num_paraphrases_per_text (int): the number of paraphrases to generate for each data.
                This parameter is for paraphrase strategy.
            tuning_steps (int): the number of steps to fine tune the classifier.
            tuning_batch_size (int): the batch size for fine tuning.
            num_sentences_to_rewrite_per_step (int): the number of data to rewrite using the
                paraphrase strategy in each tuning step. You can set is as large as the tuning
                batch size. You can also use a smaller value to speed up the tuning.
            num_updates_per_step (int): the number of classifier updates per iteration.
            period_save (int): the period in steps to save the fine tuned classifier.
        """
        raise NotImplementedError

    def model_architecture_change(self):
        pass

    def fit(self, trainset):
        """Fit the paraphrase strategy on a training set.

        Args:
            trainset (dict): a fibber dataset.
        """
        logger.info("Training is needed for this strategy. Did nothing.")

    def apply(self):
        pass
