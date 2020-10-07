import re

import numpy as np

from .measurement_base import MeasurementBase


class EditingDistance(MeasurementBase):
    """This class measures the editing distance between two sentences."""

    def __init__(self, editing_distance_ignore_punctuation=True, **kargs):
        super(EditingDistance, self).__init__()
        self._no_puctuation = editing_distance_ignore_punctuation

    def __call__(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        if self._no_puctuation:
            origin = re.sub(r"[^a-zA-Z0-9]", " ", origin)
            paraphrase = re.sub(r"[^a-zA-Z0-9]", " ", paraphrase)

        origin = origin.split()
        paraphrase = paraphrase.split()
        if len(origin) == 0 or len(paraphrase) == 0:
            return len(origin) + len(paraphrase)

        f = np.zeros((len(origin) + 1, len(paraphrase) + 1), dtype='int')
        for i in range(len(origin)):
            for j in range(len(paraphrase)):
                f[i + 1][j + 1] = min(f[i, j + 1] + 1, f[i + 1, j] + 1, f[i, j] + 1)
                if origin[i] == paraphrase[j]:
                    f[i + 1][j + 1] = min(f[i + 1][j + 1], f[i][j])

        return int(f[len(origin)][len(paraphrase)])
