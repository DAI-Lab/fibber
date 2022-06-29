from abc import ABC, abstractmethod


class MetricBase(ABC):
    """Base class for Metrics.

    All metrics should be derived from this class.

    To implement a new metric, you should at least overwrite the ``measure_example`` method.

    The simplest metric can be directly computed from a pair of text, in this case, the metric
    can use the ``origin`` and ``paraphrase`` args directly.

    Other metrics need more information from the data record. For example, ``text0``, ``text1``,
    or ``label``. Thus the ``data_record`` and ``field`` are also provided as args.

    Some metrics may run more efficiently on a batch of data. In this case, you should overwrite
    the ``measure_batch`` function. If you don't overwrite batch_call, it will compute the metric
    of paraphrase_list one by one.
    """

    def __init__(self, field, bs=32, **kwargs):
        super(MetricBase, self).__init__()
        self._field = field
        self._bs = bs

    def __repr__(self):
        return self.__class__.__name__

    def _measure_batch(self, origin, paraphrase_list, data_record=None, **kwargs):
        ret = []
        for paraphrase in paraphrase_list:
            ret.append(self.measure_example(origin, paraphrase, data_record, **kwargs))
        return ret

    def measure_batch(self, origin, paraphrase_list, data_record=None, **kwargs):
        """Measure the metric on a batch of paraphrase_list.

        If batch is larger than self._bs, the data will be split into smaller batches.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.

        Returns:
            (list): a list containing the metric for each paraphrase.
        """
        ret = []
        for i in range(0, len(paraphrase_list), self._bs):
            ret += self._measure_batch(origin,
                                       paraphrase_list[i:i + self._bs],
                                       data_record,
                                       **kwargs)
        return ret

    def _measure_multiple_examples(self, origin_list, paraphrase_list,
                                   data_record_list=None, **kwargs):
        assert len(origin_list) == len(paraphrase_list)
        ret = []
        for i in range(len(origin_list)):
            ret.append(self.measure_example(
                origin_list[i], paraphrase_list[i],
                data_record_list[i] if data_record_list is not None else None, **kwargs))
        return ret

    def measure_multiple_examples(self, origin_list, paraphrase_list,
                                  data_record_list=None, **kwargs):
        assert len(origin_list) == len(paraphrase_list)
        ret = []
        for i in range(0, len(origin_list), self._bs):
            ret += self._measure_multiple_examples(
                None if origin_list is None else origin_list[i:i + self._bs],
                paraphrase_list[i:i + self._bs],
                None if data_record_list is None else data_record_list[i:i + self._bs], **kwargs)
        return ret

    @abstractmethod
    def _measure_example(self, origin, paraphrase, data_record=None, **kwargs):
        raise NotImplementedError()

    def measure_example(self, origin, paraphrase, data_record=None, **kwargs):
        return self._measure_example(origin, paraphrase, data_record=None, **kwargs)
