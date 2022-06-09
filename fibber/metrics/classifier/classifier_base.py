import numpy as np

from fibber.metrics.metric_base import MetricBase


class ClassifierBase(MetricBase):
    """Base class for classifiers.

    All classifiers must be derived from this class.

    To implement a new classifier, you should at least overwrite the ``predict_log_dist_example``
    method. This method returns a predicted logits over classes.

    Some classifiers output label instead of distribution. In this case, you should return a
    one-hot vector.

    Some classifier may run more efficiently on a batch of data. In this case, you should overwrite
    the ``predict_log_dist_batch`` function. If you don't overwrite predict_log_dist_batch, it will
    compute the metric of paraphrase_list one by one.
    """

    def predict_log_dist_example(self, origin, paraphrase, data_record=None,
                                 field="text0"):
        """Predict the log-probability distribution over classes for one example.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            field (str): the field name to paraphrase.

        Returns:
            (np.array): a numpy array of size ``(num_labels)``.
        """
        raise NotImplementedError

    def predict_log_dist_batch(self, origin, paraphrase_list, data_record=None,
                               field="text0"):
        """Predict the log-probability distribution over classes for one batch.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            field (str): the field name to paraphrase.

        Returns:
            (np.array): a numpy array of size ``(batch_size * num_labels)``.
        """
        ret = []
        for paraphrase in paraphrase_list:
            ret.append(
                self.predict_log_dist_example(origin, paraphrase, data_record, field))
        return np.asarray(ret)

    def predict_log_dist_multiple_examples(self, origin_list, paraphrase_list,
                                           data_record_list=None, field="text0"):
        ret = []
        for i in range(len(paraphrase_list)):
            ret.append(
                self.predict_log_dist_example(
                    None if origin_list is None else origin_list[i], paraphrase_list[i],
                    data_record_list[i] if data_record_list is not None else None,
                    field))
        return np.asarray(ret)

    def predict_example(self, origin, paraphrase, data_record=None, field="text0"):
        """Predict class label for one example.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            field (str): the field name to paraphrase.

        Returns:
            (np.int): predicted label
        """
        return np.argmax(
            self.predict_log_dist_example(origin, paraphrase, data_record, field))

    def predict_batch(self, origin, paraphrase_list, data_record=None, field="text0"):
        """Predict class label for one example.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            field (str): the field name to paraphrase.

        Returns:
            (np.array): predicted label as an numpy array of size ``(batch_size)``.
        """
        return np.argmax(
            self.predict_log_dist_batch(origin, paraphrase_list, data_record, field),
            axis=1)

    def predict_multiple_examples(self, origin_list, paraphrase_list,
                                  data_record_list=None, field="text0"):
        return np.argmax(
            self.predict_log_dist_multiple_examples(origin_list, paraphrase_list,
                                                    data_record_list, field), axis=1)

    def measure_example(self, origin, paraphrase, data_record=None, field="text0"):
        """Predict class label for one example.

        Wrapper for ``predict_example``. Return type is changed from numpy.int to int.

        Args:
            origin (str): the original text.
            paraphrase (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            field (str): the field name to paraphrase.

        Returns:
            (int): predicted label
        """
        return int(self.predict_example(origin, paraphrase, data_record, field))

    def measure_batch(self, origin, paraphrase_list, data_record=None, field="text0"):
        """Predict class label for one batch.

         Wrapper for ``predict_batch``. Return type is changed from numpy.array to list of int.

         Args:
             origin (str): the original text.
             paraphrase_list (list): a set of paraphrase_list.
             data_record (dict): the corresponding data record of original text.
             field (str): the field name to paraphrase.

         Returns:
             ([int]): predicted label
         """
        return [int(x) for x in
                self.predict_batch(origin, paraphrase_list, data_record, field)]

    def measure_multiple_examples(self, origin_list, paraphrase_list,
                                  data_record_list=None, field="text0"):
        return [int(x) for x in self.predict_multiple_examples(
            origin_list, paraphrase_list, data_record_list, field)]

    def robust_tune_step(self, data_record_list):
        raise NotImplementedError

    def load_robust_tuned_model(self, desc, step):
        raise NotImplementedError

    def save_robust_tuned_model(self, desc, step):
        raise NotImplementedError
