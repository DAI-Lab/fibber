import logging

from .lm import prepare_lm
from .wordpiece_emb import get_wordpiece_emb


class AdvSampler(object):

  def __init__(self, FLAGS, trainset, testset):
    super(AdvSampler, self).__init__()

    logging.info("train language models.")
    prepare_lm(FLAGS, trainset, filter=-1)
    if FLAGS.lm_method == "adv":
      for i in range(len(trainset["label_mapping"])):
        prepare_lm(FLAGS, trainset, filter=i)

    logging.info("train word piece embeddings.")
    self._wpe = get_wordpiece_emb(FLAGS, trainset)

  def attack(self, attackset, classifier):
    pass
