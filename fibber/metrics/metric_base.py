class MetricBase(object):
    """Base class for Metrics.

    All metrics should be derived from this class.

    To implement a new metric, you should at least overwrite the `measure_example` method.

    The simplest metric can be directly computed from a pair of text, in this case, the metric
    can use the `origin` and `paraphrase` args directly.

    Other metrics need more information from the data record. For example, `text0`, `text1`, or
    `label`. Thus the `data_record` and `paraphrase_field` are also provided as args.
    """

    def __init__(self, **kargs):
        super(MetricBase, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        raise NotImplementedError()
