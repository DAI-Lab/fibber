import argparse
import logging
import data_utils

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO)

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
parser.add_argument("--clf_opt", type=str, default="adamw")
parser.add_argument("--clf_lr", type=float, default=0.00002)
parser.add_argument("--clf_decay", type=float, default=0.001)
parser.add_argument("--clf_step", type=int, default=20000)
parser.add_argument("--clf_period_summary", type=int, default=100)
parser.add_argument("--clf_period_val", type=int, default=500)
parser.add_argument("--clf_period_save", type=int, default=5000)
parser.add_argument("--clf_val_step", type=int, default=10)


# AdvSampler configs
parser.add_argument("--lm_method", choices=["full", "adv"], default="adv",
                    help=("full: fine tune LM on all data.\t"
                          "adv: fine tune LM excluding each category"))
parser.add_argument("--gibbs_round", type=int, default=2)
parser.add_argument("--gibbs_iter", type=int, default=50)
parser.add_argument("--gibbs_block", type=int, default=1)
parser.add_argument("--gibbs_eps1", type=float, default=0.98)
parser.add_argument("--gibbs_eps2", type=float, default=0.95)
parser.add_argument("--gibbs_smooth", type=float, default=1000)
parser.add_argument("--gibbs_order", choices=["seq", "rand"], default="seq")
parser.add_argument("--gibbs_topk", type=int, default=100)


def main(FLAGS):
  trainset, testset = data_utils.load_data(FLAGS.data_dir, FLAGS.dataset)
  testset_attack = data_utils.subsample_data(testset, FLAGS.n_attack)



if __name__ == "__main__":
  FLAGS = parser.parse_args()

  main(FLAGS)
