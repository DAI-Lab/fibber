import copy
import datetime
import json

import tqdm

from .. import log

logger = log.setup_custom_logger(__name__)


class StrategyBase(object):
    """docstring for Strategy."""

    def __init__(self, FLAGS):
        super(StrategyBase, self).__init__()
        # strategy config will be saved to the results.
        self._strategy_config = {
            "strategy_name": str(self)
        }

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def add_parser_args(cls, parser):
        logger.info("%s does not have any args to set from command line.", cls.__name__)

    def fit(self, trainset):
        logger.info("No tarining is used in this straget.")

    def paraphrase_example(self, data_record, field_name, n):
        raise NotImplementedError()

    def paraphrase(self, paraphrase_set, n, tmp_output_filename):
        results = copy.deepcopy(dict([(k, v) for k, v in paraphrase_set.items() if k != "data"]))
        results["strategy_config"] = self._strategy_config
        results["data"] = []

        last_tmp_output_save_time = -1

        for data_record in tqdm.tqdm(paraphrase_set["data"]):
            data_record = copy.deepcopy(data_record)
            data_record[paraphrase_set["paraphrase_field"] + "_paraphrases"] = (
                self.paraphrase_example(
                    data_record, paraphrase_set["paraphrase_field"], n))
            results["data"].append(data_record)

            # save tmp output every 30 seconds
            if datetime.datetime.now().timestamp() - last_tmp_output_save_time > 30:
                with open(tmp_output_filename, "w") as f:
                    json.dump(results, f, indent=2)
                last_tmp_output_save_time = datetime.datetime.now().timestamp()

        with open(tmp_output_filename, "w") as f:
            json.dump(results, f, indent=2)

        return results
