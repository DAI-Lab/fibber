import copy
import datetime
import json
import re

import torch
import tqdm

from fibber import log

logger = log.setup_custom_logger(__name__)

POST_PROCESSING_PATTERN = [
    (r"\s?'\s?t\s", "'t "),
    (r"\s?'\s?s\s", "'s "),
    (r"\s?'\s?ve\s", "'ve "),
    (r"\s?'\s?ll\s", "'ll "),
]


def post_process_text(text):
    for pattern in POST_PROCESSING_PATTERN:
        text = re.sub(pattern[0], pattern[1], text)
    return text


class StrategyBase(object):
    """The base class for all paraphrase strategies.

    All strategies should be derived from this class.

    The simplest way to write a strategy is to overwrite the `paraphrase_example` function. This
    function takes one data records, and returns multiple paraphrases of a given field.

    Some strategy may have hyper-parameters. Overwrite `add_parser_args` to add args to CLI. All
    args from the command line are passed as an arg to `__init__`. So you should also overwrite
    the `__init__`.

    If `__init__` is overwritten, remember to call `super().__init__(FLAGS, metric_bundle)` at the
    beginning of your `__init__`. This will initialize `self._strategy_config` and
    `self._metric_bundle`.

        `self._strategy_config` is a dict. You can store hyper-parameters in this dict. This dict
        will be saved to the result file.

        `self._metric_bundle` is the MetricBundle, you can use it to compute different metrics for
        any pairs of original and paraphrased text.

        `self._device` is a torch Device object. You can use `--strategy_gpu` to select a gpu for
        strategies. If you are using pytorch, please use this Device for computation.

    For more advanced use cases, you can overwrite the `paraphrase` function.
    """

    def __init__(self, FLAGS, metric_bundle):
        """Initialize the paraphrase_strategies.

        This function initialize the `self._strategy_config`, `self._metric_bundle` and
        `self._device`.

        If your strategy has hyper parameters, add them as args, and get those hyper parameters
        from FLAGS.

        Args:
            FLAGS: args from argparser.
            metric_bundle: a MetricBundle object.
        """
        super(StrategyBase, self).__init__()
        # paraphrase_strategies config will be saved to the results.
        self._strategy_config = {
            "strategy_name": str(self)
        }
        self._metric_bundle = metric_bundle
        if FLAGS.strategy_gpu == -1:
            logger.warning("%s is running on CPU." % str(self))
            self._device = torch.device("cpu")
        else:
            logger.info("%s metric is running on GPU %d.", str(self), FLAGS.strategy_gpu)
            self._device = torch.device("cuda:%d" % FLAGS.strategy_gpu)

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def add_parser_args(cls, parser):
        """create commandline args.

        If your strategy has hyper-parameters, add them as args to the argparser.

        Args:
            parser: an arg parser.
        """
        logger.info("%s does not have any args to set from command line.", cls.__name__)

    def fit(self, trainset):
        """Fit the paraphraser on a training set.

        Args:
            trainset (dict): a fibber dataset.
        """
        logger.info("No tarining is used in this straget.")

    def paraphrase_example(self, data_record, field_name, n):
        """Paraphrase one data record.

        This function should be overwritten by subclasses.

        Args:
            data_record (dict): a dictionary storing one data of a dataset.
            field_name (str): the field needed to be paraphrased.
            n (int): number of paraphrases.

        Returns:
            ([str]): A list contain at most n strings.
        """
        raise NotImplementedError()

    def paraphrase(self, paraphrase_set, n, tmp_output_filename):
        """Paraphrase one dataset.

        Args:
            paraphrase_set (dict): a dictionary storing one data of a dataset.
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
