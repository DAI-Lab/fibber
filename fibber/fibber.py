import numpy as np

from fibber import log
from fibber.datasets import builtin_datasets
from fibber.datasets.dataset_utils import get_dataset, verify_dataset
from fibber.metrics import MetricBundle
from fibber.paraphrase_strategies import (
    BertSamplingStrategy, IdentityStrategy, RandomStrategy, TextAttackStrategy)

logger = log.setup_custom_logger(__name__)


class Fibber(object):
    """Fibber is a unified interface for paraphrase strategies."""

    def __init__(self, arg_dict, dataset_name, strategy_name, trainset=None, testset=None,
                 output_dir=".", bert_clf_steps=5000):
        """Initialize

        Args:
            arg_dict (dict): a dict of hyper parameters for the MetricBundle and strategy.
            dataset_name (str): the name of the dataset.
            strategy_name (str): the strategy name.
            trainset (str): fibber dataset.
            testset (str): fibber testset.
            output_dir (str): directory to cache the strategy.
        """
        super(Fibber, self).__init__()

        # setup dataset
        if dataset_name in builtin_datasets:
            if trainset is not None or testset is not None:
                logger.error(("dataset name %d conflict with builtin dataset. "
                              "set trainset and testset to None.") % dataset_name)
                raise RuntimeError
            trainset, testset = get_dataset(dataset_name)
        else:
            verify_dataset(trainset)
            verify_dataset(testset)

        self._metric_bundle = MetricBundle(
            enable_bert_clf_prediction=True,
            use_gpu_id=arg_dict["use_gpu_id"],
            gpt2_gpu_id=arg_dict["gpt2_gpu_id"],
            bert_gpu_id=arg_dict["bert_gpu_id"],
            dataset_name=dataset_name,
            trainset=trainset, testset=testset,
            bert_clf_steps=bert_clf_steps)

        strategy_gpu_id = arg_dict["strategy_gpu_id"]
        if strategy_name == "RandomStrategy":
            self._strategy = RandomStrategy(
                arg_dict, dataset_name, strategy_gpu_id, output_dir, self._metric_bundle)
        if strategy_name == "IdentityStrategy":
            self._strategy = IdentityStrategy(
                arg_dict, dataset_name, strategy_gpu_id, output_dir, self._metric_bundle)
        if strategy_name == "TextAttackStrategy":
            self._strategy = TextAttackStrategy(
                arg_dict, dataset_name, strategy_gpu_id, output_dir, self._metric_bundle)
        if strategy_name == "BertSamplingStrategy":
            self._strategy = BertSamplingStrategy(
                arg_dict, dataset_name, strategy_gpu_id, output_dir, self._metric_bundle)
        if self._strategy is None:
            logger.error("unknown strategy name %s." % strategy_name)
            raise RuntimeError

        self._strategy.fit(trainset)

        self._trainset = trainset
        self._testset = testset

    def paraphrase(self, data_record, field_name="text0", n=20):
        """Paraphrase a given data record.

        Args:
            data_record (dict): data record to be paraphrased.
            field_name (str): select from ``["text0", "text1"]``
            n (int): number of paraphrases.

        Returns:
            * a list of str as paraphrased sentences.
            * a list of dict as corresponding metrics.
        """
        paraphrases = self._strategy.paraphrase_example(data_record, field_name, n)
        metrics = []
        for item in paraphrases:
            metrics.append(self._metric_bundle.measure_example(
                data_record[field_name], item, data_record, field_name))
        return data_record[field_name], paraphrases, metrics

    def paraphrase_a_random_sentence(self, n=20, from_testset=True):
        """Randomly pick one data, then paraphrase it.

        Args:
            n (int): number of paraphrases.
            from_testset (bool): if true, select data from test set, otherwise from training set.

        Returns:
            * a str as the original text.
            * a list of str as the paraphrased text.
            * a list of dict as corresponding metrics.
        """
        dataset = self._testset if from_testset else self._trainset
        field = dataset["paraphrase_field"]

        data_record = np.random.choice(dataset["data"])

        paraphrases, metrics = self.paraphrase(data_record, field_name=field, n=n)

        return data_record[field], paraphrases, metrics

    def get_metric_bundle(self):
        """"Get the metric bundle."""
        return self._metric_bundle
