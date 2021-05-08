import argparse
import copy
import subprocess

COMMON_CONFIG = {
    "--subsample_testset": 1000,
    "--num_paraphrases_per_text": 50,
    "--robust_tuning": "0",
    # ignored when robut_tuning is 0 and load_robust_tuned_clf is not set
    "--robust_tuning_steps": 5000,
    "--use_sbert": "1"
}

GPU_CONFIG = {
    "single": {
        "--bert_gpu_id": 0,
        "--use_gpu_id": 0,
        "--gpt2_gpu_id": 0,
        "--strategy_gpu_id": 0,
        "--sbert_gpu_id": 0
    },
    "multi": {
        "--bert_gpu_id": 0,
        "--use_gpu_id": 0,
        "--gpt2_gpu_id": 1,
        "--sbert_gpu_id": 1,
        "--strategy_gpu_id": 2,
    }
}

DATASET_CONFIG = {
    "ag_no_title": {
        "--dataset": "ag_no_title",
        "--output_dir": "exp-ag_no_title",
        "--bert_clf_steps": 20000
    },
    "mr": {
        "--dataset": "mr",
        "--output_dir": "exp-mr",
        "--bert_clf_steps": 5000
    },
    "imdb": {
        "--dataset": "imdb",
        "--output_dir": "exp-imdb",
        "--bert_clf_steps": 5000
    },
    "yelp": {
        "--dataset": "yelp",
        "--output_dir": "exp-yelp",
        "--bert_clf_steps": 20000
    },
    "snli": {
        "--dataset": "snli",
        "--output_dir": "exp-snli",
        "--bert_clf_steps": 20000
    },
    "mnli": {
        "--dataset": "mnli",
        "--output_dir": "exp-mnli",
        "--bert_clf_steps": 20000
    },
    "mnli_mis": {
        "--dataset": "mnli_mis",
        "--output_dir": "exp-mnli_mis",
        "--bert_clf_steps": 20000
    },
}

STRATEGY_CONFIG = {
    "identity": {
        "--strategy": "IdentityStrategy"
    },
    "random": {
        "--strategy": "RandomStrategy"
    },
    "textfooler": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "TextFoolerJin2019"
    },
    "pso": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "PSOZang2020"
    },
    "bertattack": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "BERTAttackLi2020"
    },
    "bae": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "BAEGarg2019"
    },
    "asrs": {
        "--strategy": "BertSamplingStrategy",
        "--bs_enforcing_dist": "wpe",
        "--bs_wpe_threshold": 1.0,
        "--bs_wpe_weight": 1000,
        "--bs_use_threshold": 0.95,
        "--bs_use_weight": 500,
        "--bs_gpt2_weight": 5,
        "--bs_sampling_steps": 200,
        "--bs_burnin_steps": 100,
        "--bs_clf_weight": 3,
        "--bs_window_size": 3,
        "--bs_accept_criteria": "joint_weighted_criteria",
        "--bs_burnin_enforcing_schedule": "1",
        "--bs_burnin_criteria_schedule": "1",
        "--bs_seed_option": "origin",
        "--bs_split_sentence": "auto",
        "--bs_lm_option": "finetune",
        "--bs_stanza_port": 9001,
        "--bs_similarity_metric": "SBERTSemanticSimilarityMetric",
    },
    "asrs-no_enf": {
        "--bs_enforcing_dist": "none",
    },
    "asrs-no_dec": {
        "--bs_accept_criteria": "all",
    },
    "asrs-no_clf": {
        "--bs_clf_weight": 0,
    },
    "asrs-no_block": {
        "--bs_window_size": 1,
    },
    "asrs-nli": {
        "--bs_use_weight": 100,
        "--bs_gpt2_weight": 3,
    },
    "nabs": {
        "--strategy": "NonAutoregressiveBertSamplingStrategy",
        "--nabs_lm_steps": 40000,
        "--nabs_enforce_similarity": "1"
    },
    "narrl": {
        "--strategy": "NARRLStrategy",
        "--nr_lm_steps": 40000,
    }
}


def to_command(args):
    ret = []
    for k, v in args.items():
        ret.append(k)
        ret.append(str(v))

    return ret


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", choices=["single", "multi"], default="single")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    parser.add_argument("--strategy", choices=list(STRATEGY_CONFIG.keys()) + ["all"],
                        default="all")
    parser.add_argument("--robust_desc", type=str, default=None)

    args = parser.parse_args()
    if args.dataset == "all":
        dataset_list = list(DATASET_CONFIG.keys())
    else:
        dataset_list = [args.dataset]

    if args.strategy == "all":
        strategy_list = list(STRATEGY_CONFIG.keys())
    else:
        strategy_list = [args.strategy]

    for dataset in dataset_list:
        for strategy in strategy_list:
            command = ["python3", "-m", "fibber.benchmark.benchmark"]
            if args.robust_desc is not None:
                command += to_command({"--load_robust_tuned_clf": args.robust_desc})
            command += to_command(COMMON_CONFIG)
            command += to_command(GPU_CONFIG[args.gpu])
            command += to_command(DATASET_CONFIG[dataset])
            if strategy[:4] == "asrs":
                strategy_config_tmp = copy.copy(STRATEGY_CONFIG["asrs"])
                if strategy != "asrs":
                    for k, v in STRATEGY_CONFIG[strategy].items():
                        strategy_config_tmp[k] = v
                command += to_command(strategy_config_tmp)
            else:
                command += to_command(STRATEGY_CONFIG[strategy])
            subprocess.call(command)


if __name__ == "__main__":
    main()
