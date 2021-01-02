import numpy as np

from fibber.metrics.metric_base import MetricBase


class ClassifierBase(MetricBase):
    """Base class for classifiers.

    All classifiers must be derived from this class.

    To implement a new classifier, you should at least overwrite the ``predict_dist_example``
    method. This method returns a predicted logits over classes.

    Some classifiers output label instead of distribution. In this case, you should return a
    one-hot vector.

    Some classifier may run more efficiently on a batch of data. In this case, you should overwrite
    the ``predict_dist_batch`` function. If you don't overwrite predict_dist_batch, it will compute
    the metric of paraphrase_list one by one.
    """

    def predict_dist_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Predict the log-probability distribution over classes for one example.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (np.array): a numpy array of size ``(num_labels)``.
        """
        raise NotImplementedError

    def predict_dist_batch(self, origin, paraphrase_list, data_record=None,
                           paraphrase_field="text0"):
        """Predict the log-probability distribution over classes for one batch.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (np.array): a numpy array of size ``(batch_size * num_labels)``.
        """
        ret = []
        for paraphrase in paraphrase_list:
            ret.append(
                self.predict_dist_example(origin, paraphrase, data_record, paraphrase_field))
        return np.asarray(ret)

    def predict_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Predict class label for one example.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (np.int): predicted label
        """
        return np.argmax(
            self.predict_dist_example(origin, paraphrase, data_record, paraphrase_field))

    def predict_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0"):
        """Predict class label for one example.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (np.array): predicted label as an numpy array of size ``(batch_size)``.
        """
        return np.argmax(
            self.predict_dist_batch(origin, paraphrase_list, data_record, paraphrase_field),
            axis=1)

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Predict class label for one example.

        Wrapper for ``predict_example``. Return type is changed from numpy.int to int.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (int): predicted label
        """
        return int(self.predict_example(origin, paraphrase, data_record, paraphrase_field))

    def measure_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0"):
        """Predict class label for one batch.

         Wrapper for ``predict_batch``. Return type is changed from numpy.array to list of int.

         Args:
             origin (str): the original text.
             paraphrase_list (list): a set of paraphrase_list.
             data_record (dict): the corresponding data record of original text.
             paraphrase_field (str): the field name to paraphrase.

         Returns:
             ([int]): predicted label
         """
        return [int(x) for x in
                self.predict_batch(origin, paraphrase_list, data_record, paraphrase_field)]
