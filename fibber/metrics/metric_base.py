class MetricBase(object):
    """Base class for Metrics.

    All metrics should be derived from this class.

    To implement a new metric, you should at least overwrite the ``measure_example`` method.

    The simplest metric can be directly computed from a pair of text, in this case, the metric
    can use the ``origin`` and ``paraphrase`` args directly.

    Other metrics need more information from the data record. For example, ``text0``, ``text1``,
    or ``label``. Thus the ``data_record`` and ``paraphrase_field`` are also provided as args.

    Some metrics may run more efficiently on a batch of data. In this case, you should overwrite
    the ``measure_batch`` function. If you don't overwrite batch_call, it will compute the metric
    of paraphrase_list one by one.
    """

    def __init__(self, **kargs):
        super(MetricBase, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def measure_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0"):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (list): a list containing the metric for each paraphrase.
        """
        ret = []
        for paraphrase in paraphrase_list:
            ret.append(self.measure_example(origin, paraphrase, data_record, paraphrase_field))
        return ret

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        raise NotImplementedError()
