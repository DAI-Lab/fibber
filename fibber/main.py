import argparse
import json
import os

from . import classifier, data_utils, log
from .attack.advsampler import AdvSampler
from .evaluate.evaluation import make_measures, evaluate

logger = log.setup_custom_logger('root')


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="data/",
                    help="directory to all datasets.")
parser.add_argument("--dataset", default="ag",
                    choices=["ag", "yelp", "mr", "imdb", "snli",
                             "mnli", "mnli_mis"])
parser.add_argument("--n_attack", type=int, default=1000,
                    help="number of sentences to attack (sample from testset).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="directory to save model and logs.")
parser.add_argument("--leaderboard", type=str, default=None,
                    help="leaderboard filename.")

# Classifier configs
parser.add_argument("--clf_bs", type=int, default=32)
parser.add_argument("--clf_opt", type=str, default="adamw")
parser.add_argument("--clf_lr", type=float, default=0.00002)
parser.add_argument("--clf_decay", type=float, default=0.001)
parser.add_argument("--clf_step", type=int, default=20000)
parser.add_argument("--clf_period_summary", type=int, default=100)
parser.add_argument("--clf_period_val", type=int, default=500)
parser.add_argument("--clf_period_save", type=int, default=5000)
parser.add_argument("--clf_val_step", type=int, default=10)

# LM configs
parser.add_argument("--lm_bs", type=int, default=32)
parser.add_argument("--lm_opt", type=str, default="adamw")
parser.add_argument("--lm_lr", type=float, default=0.0001)
parser.add_argument("--lm_decay", type=float, default=0.01)
parser.add_argument("--lm_step", type=int, default=10000)
parser.add_argument("--lm_period_summary", type=int, default=100)
parser.add_argument("--lm_period_save", type=int, default=5000)

# WPE configs
parser.add_argument("--wpe_bs", type=int, default=5000)
parser.add_argument("--wpe_lr", type=float, default=1)
parser.add_argument("--wpe_momentum", type=float, default=0)
parser.add_argument("--wpe_step", type=int, default=2000)
parser.add_argument("--wpe_peroid_lr_halve", type=int, default=1000)


# AdvSampler configs
parser.add_argument("--attack_method", choices=["all", "adv"], default="adv",
                    help=("all: fine tune LM on all data.\t"
                          "adv: fine tune LM excluding each category"))
parser.add_argument("--gibbs_max_len", type=int, default=200)
parser.add_argument("--gibbs_round", type=int, default=2)
parser.add_argument("--gibbs_iter", type=int, default=50)
parser.add_argument("--gibbs_block", type=int, default=1)
parser.add_argument("--gibbs_eps1", type=float, default=0.98)
parser.add_argument("--gibbs_eps2", type=float, default=0.95)
parser.add_argument("--gibbs_smooth", type=float, default=1000)
parser.add_argument("--gibbs_order", choices=["seq", "rand"], default="seq")
parser.add_argument("--gibbs_topk", type=int, default=100)
parser.add_argument("--gibbs_keep_entity", choices=["0", "1"], default="1")


def main(FLAGS):
  logger.info("all flags: %s", FLAGS)
  with open(FLAGS.output_dir + "/config.json", "w") as f:
    json.dump(vars(FLAGS), f, indent=2)
  trainset, testset = data_utils.load_data(FLAGS.data_dir, FLAGS.dataset)
  attackset = data_utils.subsample_data(testset, FLAGS.n_attack)

  logger.info("train classifier")
  clf_model = classifier.get_clf(FLAGS, trainset, testset)

  logger.info("prepare attacker")
  attacker = AdvSampler(FLAGS, trainset, testset)
  logger.info("attack")
  name, result = attacker.attack_clf(FLAGS, attackset, clf_model)
  del attacker

  logger.info("evaluation")
  measures = make_measures("data/", use_editing=True, use_use=True)
  if "nli" in FLAGS.dataset:
    measure_sentence = "s1"
  else:
    measure_sentence = "s0"

  for item in result:
    item["eval"] = evaluate(item["ori"][measure_sentence],
                            item["adv"][measure_sentence],
                            measures)

  with open(FLAGS.output_dir + "/" + name + ".json", "w") as f:
    json.dump(result, f, indent=2)

if __name__ == "__main__":
  FLAGS = parser.parse_args()
  assert FLAGS.output_dir is not None
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  main(FLAGS)
