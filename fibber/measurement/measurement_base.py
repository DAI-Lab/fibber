class MeasurementBase(object):
    """docstring for MeasurementBase."""

    def __init__(self, **kargs):
        super(MeasurementBase, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, origin, paraphrase):
        raise NotImplementedError()
