class TuningStrategyBase(object):
    """Base class for Tuning strategy"""

    def __repr__(self):
        return self.__class__.__name__

    def fine_tune_classifier(self,
                             metric_bundle,
                             paraphrase_strategy,
                             train_set,
                             num_paraphrases_per_text,
                             tuning_steps,
                             tuning_batch_size=32,
                             num_sentences_to_rewrite_per_step=20,
                             num_updates_per_step=5,
                             period_save=1000):
        """Fine tune the classifier using given data.

        Args:
            metric_bundle (MetricBundle): a metric bundle including the target classifier.
            paraphrase_strategy (StrategyBase): a paraphrase strategy to fine tune the classifier.
            train_set (dict): the training set of the classifier.
            num_paraphrases_per_text (int): the number of paraphrases to generate for each data.
                This parameter is for paraphrase strategy.
            tuning_steps (int): the number of steps to fine tune the classifier.
            tuning_batch_size (int): the batch size for fine tuning.
            num_sentences_to_rewrite_per_step (int): the number of data to rewrite using the
                paraphrase strategy in each tuning step. You can set is as large as the tuning
                batch size. You can also use a smaller value to speed up the tuning.
            num_updates_per_step (int): the number of classifier updates per iteration.
            period_save (int): the period in steps to save the fine tuned classifier.
        """
        raise NotImplementedError
