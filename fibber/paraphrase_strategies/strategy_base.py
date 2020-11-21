import copy
import datetime
import json

import torch
import tqdm

from fibber import log

logger = log.setup_custom_logger(__name__)


class StrategyBase(object):
    """The base class for all paraphrase strategies.

    The simplest way to write a strategy is to overwrite the ``paraphrase_example`` function. This
    function takes one data records, and returns multiple paraphrases of a given field.

    For more advanced use cases, you can overwrite the ``paraphrase`` function.

    Some strategy may have hyper-parameters. Add hyper parameters into the class attribute
    ``__hyperparameters__``.

    Hyperparameters defined in ``__hyperparameters__`` can be added to the command line arg parser
    by ``add_parser_args(parser)``. The value of the hyperparameters will be added to
    ``self._strategy_config``.

    Attributes:
        __abbr__ (str): a unique string as an abbreviation for the strategy.
        __hyper_parameters__ (list): A list of tuples that defines the hyperparameters for the
            strategy. Each tuple is ``(name, type, default, help)``. For example::

                __hyperparameters = [ ("p1", int, -1, "the first hyper parameter"), ...]
    """
    __abbr__ = "base"
    __hyperparameters__ = []

    def __init__(self, arg_dict, dataset_name, strategy_gpu_id, output_dir, metric_bundle):
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
            output_dir (str): a directory to save any models or temporary files.
            metric_bundle (MetricBundle): a MetricBundle object.
        """
        super(StrategyBase, self).__init__()

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

        self._output_dir = output_dir
        self._dataset_name = dataset_name

    def __repr__(self):
        return self.__class__.__name__

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

    def fit(self, trainset):
        """Fit the paraphrase strategy on a training set.

        Args:
            trainset (dict): a fibber dataset.
        """
        logger.info("Training is needed for this strategy. Did nothing.")

    def paraphrase_example(self, data_record, field_name, n):
        """Paraphrase one data record.

        This function should be overwritten by subclasses. When overwriting this class, you can
        use ``self._strategy_config``, ``self._metric_bundle``,  ``self._device``,
        ``self._output_dir``, and ``self._dataset_name``

        Args:
            data_record (dict): a dict storing one data of a dataset.
            field_name (str): the field needed to be paraphrased.
            n (int): number of paraphrases.

        Returns:
            ([str,]): A list contain at most n strings.
        """
        raise NotImplementedError()

    def paraphrase_dataset(self, paraphrase_set, n, tmp_output_filename):
        """Paraphrase one dataset.

        Args:
            paraphrase_set (dict): a dict storing one data of a dataset.
            n (int): number of paraphrases.
            tmp_output_filename (str): the output json filename to save results during running.

        Returns:
            (dict): A dict containing the original text and paraphrased text.
        """
        results = copy.deepcopy(dict([(k, v) for k, v in paraphrase_set.items() if k != "data"]))
        results["strategy_config"] = self._strategy_config
        results["data"] = []

        last_tmp_output_save_time = -1

        for data_record in tqdm.tqdm(paraphrase_set["data"]):
            data_record = copy.deepcopy(data_record)
            data_record[paraphrase_set["paraphrase_field"] + "_paraphrases"] = (
                self.paraphrase_example(
                    data_record, paraphrase_set["paraphrase_field"], n)[:n])
            results["data"].append(data_record)

            # save tmp output every 30 seconds
            if datetime.datetime.now().timestamp() - last_tmp_output_save_time > 30:
                with open(tmp_output_filename, "w") as f:
                    json.dump(results, f, indent=2)
                last_tmp_output_save_time = datetime.datetime.now().timestamp()

        with open(tmp_output_filename, "w") as f:
            json.dump(results, f, indent=2)

        return results
