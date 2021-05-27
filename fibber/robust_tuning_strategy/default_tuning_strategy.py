import copy
import datetime

import numpy as np
import tqdm
from fibber import log


logger = log.setup_custom_logger(__name__)


class TuningStrategyBase(object):
    pass


class DefaultTuningStrategy(TuningStrategyBase):
    """Default Tuning Strategy tunes the target classifier using adversarial training.

    This strategy maintains two lists of data. The ``correct_list`` contains the examples that are
    correctly classified by the classifer. The ``incorrect_list`` contains the examples that are
    misclassified. The training set is added to the ``correct_list`` at beginning.

    In each tuning step, this strategy select some sentences from the training set to rewrite. The
    rewritten sentences are added to two lists correspondingly.

    Then we sample half of tuning batch from ``correct_list`` and half from ``incorrect_list``.
    The selected batch is used to tune the classifier. After tuning, these selected examples are
    put back to two lists according to the current prediction of the classifier.
    """

    def __init__(self, seed=1234):
        """Initialize the strategy."""
        super(DefaultTuningStrategy, self).__init__()
        self._rng = np.random.RandomState(seed)

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
        exp_time = datetime.datetime.now().strftime("%m%d%H%M")
        global_step = 0

        classifier = metric_bundle.get_target_classifier()
        classifier.robust_tune_init("adamw", 0.00002, 0.001, tuning_steps * num_updates_per_step)

        paraphrase_field = train_set["paraphrase_field"]

        pbar = tqdm.tqdm(total=tuning_steps)

        data_record_list = copy.deepcopy(train_set["data"])
        correct_set = set([i for i in range(len(data_record_list))])
        incorrect_set = set()

        while True:
            global_step += 1

            # use paraphrase strategy to augment training data.
            for i in range(num_sentences_to_rewrite_per_step):
                data_record_t = self._rng.choice(train_set["data"])
                paraphrase_list = paraphrase_strategy.paraphrase_example(
                    data_record_t,
                    paraphrase_field,
                    num_paraphrases_per_text)

                if i == 0:
                    logger.info(paraphrase_list[0])

                predict_label_list = classifier.measure_batch(
                    data_record_t[paraphrase_field], paraphrase_list, data_record_t,
                    paraphrase_field)

                for (paraphrase, predict_label) in zip(paraphrase_list, predict_label_list):

                    data_record_new = copy.deepcopy(data_record_t)
                    data_record_new[paraphrase_field] = paraphrase
                    data_record_list.append(data_record_new)

                    if predict_label != data_record_t["label"]:
                        incorrect_set.add(len(data_record_list) - 1)
                    # else:
                    #     correct_set.add(len(data_record_list) - 1)1

            correct_cnt = 0
            tot_cnt = 0
            loss = 0
            for i in range(num_updates_per_step):
                # sample sentences from the list.
                tuning_batch_id_list = []
                if len(incorrect_set) > 0:
                    n_incorrect = min(tuning_batch_size // 2, len(incorrect_set))
                    tuning_batch_id_t = self._rng.choice(
                        tuple(incorrect_set), n_incorrect, replace=False)
                    for idx in tuning_batch_id_t:
                        incorrect_set.remove(idx)
                    tuning_batch_id_list += list(tuning_batch_id_t)
                else:
                    n_incorrect = 0
                    continue

                if len(correct_set) > 0:
                    tuning_batch_id_t = self._rng.choice(
                        tuple(correct_set), min(tuning_batch_size // 2, len(correct_set)),
                        replace=False)

                    # for idx in tuning_batch_id_t:
                    #     correct_set.remove(idx)
                    tuning_batch_id_list += list(tuning_batch_id_t)

                # fine-tune the classifier
                predict_list, loss = classifier.robust_tune_step(
                    [data_record_list[idx] for idx in tuning_batch_id_list])

                for id_, (idx, predict) in enumerate(zip(tuning_batch_id_list, predict_list)):
                    if predict != data_record_list[idx]["label"]:
                        if id_ < n_incorrect:
                            incorrect_set.add(idx)
                    else:
                        correct_cnt += 1
                    tot_cnt += 1

            pbar.update(1)
            pbar.set_postfix({
                "loss": loss,
                "training_acc": correct_cnt / tot_cnt if tot_cnt > 0 else -1,
                "incorrect_list": len(incorrect_set),
                "correct_list": len(correct_set)
            })

            if global_step % period_save == 0 or global_step == tuning_steps:
                classifier.save_robust_tuned_model(
                    self.__repr__() + "-" + str(num_sentences_to_rewrite_per_step)
                    + "-" + str(paraphrase_strategy)
                    + "-" + exp_time,
                    global_step)

            if global_step >= tuning_steps:
                break

        pbar.close()
