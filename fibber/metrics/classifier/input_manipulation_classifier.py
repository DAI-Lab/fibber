import numpy as np
from scipy.special import log_softmax as scipy_log_softmax

from fibber.metrics.classifier.classifier_base import ClassifierBase


def hard_agg(x):
    ret = np.zeros(x.shape[1])
    for idx in np.argmax(x, axis=1):
        ret[idx] += 1
    return scipy_log_softmax(ret)


def soft_agg(x):
    return scipy_log_softmax(np.mean(x, axis=0))


class InputManipulationClassifier(ClassifierBase):
    def __init__(self, original_classifier, input_manipulation, name, agg="hard", **kwargs):
        super(InputManipulationClassifier, self).__init__(**kwargs)
        if agg == "hard":
            self._agg_fn = hard_agg
        elif agg == "soft":
            self._agg_fn = soft_agg
        else:
            raise RuntimeError("agg not supported")
        self._classifier = original_classifier
        self._input_manipulation = input_manipulation
        self._name = name

    def __str__(self):
        return self._name

    def _predict_log_dist_example(self, origin, paraphrase, data_record=None):
        paraphrase_list = self._input_manipulation(
            [paraphrase], [data_record] if data_record is not None else None)[0]
        return self._agg_fn(self._classifier.predict_log_dist_batch(
            origin, paraphrase_list, data_record))

    def _predict_log_dist_batch(self, origin, paraphrase_list, data_record=None):
        paraphrase_list = self._input_manipulation(
            paraphrase_list,
            [data_record] * len(paraphrase_list) if data_record is not None else None)

        paraphrase_list_cat = []
        for item in paraphrase_list:
            paraphrase_list_cat += item
        logits = self._classifier.predict_log_dist_batch(origin, paraphrase_list_cat, data_record)

        p = 0
        logits_reformat = []
        for item in paraphrase_list:
            logits_reformat.append(logits[p:p + len(item)])
            p += len(item)
        assert p == len(logits)

        ret = []
        for item in logits_reformat:
            ret.append(self._agg_fn(item))
        return np.asarray(ret)

    def _predict_log_dist_multiple_examples(self, origin_list, paraphrase_list,
                                            data_record_list=None):
        paraphrase_list = self._input_manipulation(
            paraphrase_list, data_record_list if data_record_list is not None else None)

        paraphrase_list_cat = []
        origin_list_cat = []
        data_record_list_cat = []
        for idx, item in enumerate(paraphrase_list):
            paraphrase_list_cat += item
            if origin_list is not None:
                origin_list_cat += [origin_list[idx]] * len(item)
            if data_record_list is not None:
                data_record_list_cat += [data_record_list_cat[idx]] * len(item)
        logits = self._classifier.predict_log_dist_multiple_examples(
            None if origin_list is None else origin_list_cat,
            paraphrase_list_cat,
            None if data_record_list is None else data_record_list_cat)

        p = 0
        logits_reformat = []
        for item in paraphrase_list:
            logits_reformat.append(logits[p:p + len(item)])
            p += len(item)
        assert p == len(logits)

        ret = []
        for item in logits_reformat:
            ret.append(self._agg_fn(item))
        return np.asarray(ret)

    def robust_tune_init(self, optimizer, lr, weight_decay, steps):
        raise NotImplementedError

    def robust_tune_step(self, data_record_list):
        raise NotImplementedError

    def load_robust_tuned_model(self, save_path):
        raise NotImplementedError

    def save_robust_tuned_model(self, load_path):
        raise NotImplementedError
