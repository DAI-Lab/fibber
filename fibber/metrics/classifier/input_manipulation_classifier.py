from fibber.metrics.classifier.classifier_base import ClassifierBase


class InputManipulationClassifier(ClassifierBase):
    def __init__(self, original_classifier, input_manipulation, name):
        super(InputManipulationClassifier, self).__init__()
        self._classifier = original_classifier
        self._input_manipulation = input_manipulation
        self._name = name

    def __str__(self):
        return self._name

    def predict_log_dist_example(self, origin, paraphrase, data_record=None,
                                 field="text0"):
        paraphrase = self._input_manipulation(
            [paraphrase], [data_record] if data_record is not None else None)[0]
        return self._classifier.predict_log_dist_example(
            origin, paraphrase, data_record, field)

    def predict_log_dist_batch(self, origin, paraphrase_list, data_record=None,
                               field="text0"):
        paraphrase_list = self._input_manipulation(
            paraphrase_list,
            [data_record] * len(paraphrase_list) if data_record is not None else None)
        return self._classifier.predict_log_dist_batch(
            origin, paraphrase_list, data_record, field)

    def predict_log_dist_multiple_examples(self, origin_list, paraphrase_list,
                                           data_record_list=None, field="text0"):
        paraphrase_list = self._input_manipulation(
            paraphrase_list, data_record_list if data_record_list is not None else None)
        return self._classifier.predict_log_dist_multiple_examples(
            origin_list, paraphrase_list, data_record_list, field)
