from .lm import train_lm

def prepare_lm(FLAGS, trainset, testset):
  train_lm(FLAGS, trainset, filter=-1)
  if FLAGS.lm_method == "adv":
    for i in range(len(trainset["label_mapping"])):
      train_lm(FLAGS, trainset, filter=i)
