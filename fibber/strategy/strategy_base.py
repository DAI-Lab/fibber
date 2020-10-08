import copy
import datetime
import json

import tqdm

from .. import log

logger = log.setup_custom_logger(__name__)


class StrategyBase(object):
    """The base class for all paraphrase strategies.

    All strategies should be derived from this class. To implement a strategy, please implement
    the following functions.

    - Add command line args to `add_parser_args` if necessary.
    - Config the parser in `__init__` if necessary. Add configuration details to
      `self._strategy_config`.
    - Overwrite fit if the strategy needs to train on the training data.
    - Overwrite paraphrase_example to paraphrase one example.
    """

    def __init__(self, FLAGS):
        """Initialize the strategy."""
        super(StrategyBase, self).__init__()
        # strategy config will be saved to the results.
        self._strategy_config = {
            "strategy_name": str(self)
        }

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def add_parser_args(cls, parser):
        """create commandline args.

        Args:
            parser: an arg parser.
        """
        logger.info("%s does not have any args to set from command line.", cls.__name__)

    def fit(self, trainset):
        """Fit the paraphraser on a training set."""
        logger.info("No tarining is used in this straget.")

    def paraphrase_example(self, data_record, field_name, n):
        """Paraphrase one data record.

        This function should be overwritten by subclasses.

        Args:
            data_record: a dictionary storing one data of a dataset.
            field_name: the field needed to be paraphrased.
            n: number of paraphrases.

        Returns:
            A list contain at most n strings.
        """
        raise NotImplementedError()

    def paraphrase(self, paraphrase_set, n, tmp_output_filename):
        """Paraphrase one datasset.

        Args:
            data_record: a dictionary storing one data of a dataset.
            field_name: the field needed to be paraphrased.
            n: number of paraphrases.
            tmp_output_filename: the output json filename to save results during running.

        Returns:
            A dictionary including paraphrase_set and paraphrases for each data record.
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
