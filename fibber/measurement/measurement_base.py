class MeasurementBase(object):
    """Base class for measurements."""

    def __init__(self, **kargs):
        super(MeasurementBase, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        raise NotImplementedError()
