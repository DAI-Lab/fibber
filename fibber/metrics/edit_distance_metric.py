"""This metric outputs the word-level editing distance between original text and paraphrased
text.
"""

import re

import numpy as np

from fibber.metrics.metric_base import MetricBase


class EditDistanceMetric(MetricBase):
    """This class measures the editing distance between two sentences."""

    def __init__(self, editing_distance_ignore_punctuation=True, **kargs):
        """Initialize.

        Args:
            editing_distance_ignore_punctuation (bool): whether to ignore punctuation when
                computing editing distance.
        """
        super(EditDistanceMetric, self).__init__()
        self._no_puctuation = editing_distance_ignore_punctuation

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """compute editing distance between original and parapharse.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.

        Returns:
            (int): the editing distance.
        """
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
