import datetime

import numpy as np
import tqdm

from fibber import log
from fibber.defense_strategies.defense_strategy_base import DefenseStrategyBase

logger = log.setup_custom_logger(__name__)


class NaiveDefenseStrategy(DefenseStrategyBase):
    def __init__(self, seed=1234):
        """Initialize the strategy."""
        super(NaiveDefenseStrategy, self).__init__()
        self._rng = np.random.RandomState(seed)

    def fine_tune_classifier(self,
                             metric_bundle,
                             paraphrase_strategy,
                             train_set,
                             num_paraphrases_per_text,
                             tuning_steps,
                             tuning_batch_size=64,
                             num_sentences_to_rewrite_per_step=16,
                             num_updates_per_step=1,
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
        assert num_updates_per_step == 1
        assert num_sentences_to_rewrite_per_step * 2 == tuning_batch_size

        exp_time = datetime.datetime.now().strftime("%m%d%H%M")
        global_step = 0

        classifier = metric_bundle.get_target_classifier()
        classifier.robust_tune_init("adamw", 0.00002, 0.001, tuning_steps * num_updates_per_step)

        paraphrase_field = train_set["paraphrase_field"]

        pbar = tqdm.tqdm(total=tuning_steps)

        while True:
            global_step += 1

            # use paraphrase strategy to augment training data.

            # data_pool_1 = []
            # data_pool_2 = []
            #
            # for i in range(num_sentences_to_rewrite_per_step):
            #     data_record_t = self._rng.choice(train_set["data"])
            #     paraphrase_list, _ = paraphrase_strategy.paraphrase_example(
            #         data_record_t,
            #         paraphrase_field,
            #         num_paraphrases_per_text)
            #
            #     if i == 0:
            #         logger.info(paraphrase_list[0])
            #
            #     predict_label_list = classifier.measure_batch(
            #         data_record_t[paraphrase_field], paraphrase_list, data_record_t,
            #         paraphrase_field)
            #
            #     for (paraphrase, predict_label) in zip(paraphrase_list, predict_label_list):
            #         if predict_label != data_record_t["label"]:
            #             data_record_new = copy.deepcopy(data_record_t)
            #             data_record_new[paraphrase_field] = paraphrase
            #             data_pool_2.append(data_record_new)
            #     data_pool_1.append(data_record_t)
            data_pool_1 = list(
                self._rng.choice(train_set["data"], num_sentences_to_rewrite_per_step))
            paraphrases = paraphrase_strategy.paraphrase_multiple_examples(
                data_pool_1, paraphrase_field)
            data_pool_2 = [item.copy() for item in data_pool_1]
            for idx in range(num_sentences_to_rewrite_per_step):
                data_pool_2[idx][paraphrase_field] = paraphrases[idx]

            correct_cnt = 0
            tot_cnt = 0
            loss = 0
            for i in range(num_updates_per_step):
                # fine-tune the classifier
                batch_data_records = data_pool_1 + data_pool_2
                predict_list, loss = classifier.robust_tune_step(batch_data_records)

                for pred, data_record in zip(predict_list, batch_data_records):
                    if pred == data_record["label"]:
                        correct_cnt += 1
                    tot_cnt += 1
            pbar.update(1)
            pbar.set_postfix({
                "loss": loss,
                "training_acc": correct_cnt / tot_cnt if tot_cnt > 0 else -1,
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
