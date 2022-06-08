import argparse
import copy
import subprocess

COMMON_CONFIG = {
    "--subsample_testset": 500,
    "--max_paraphrases": 20,
}

GPU_CONFIG = {
    "single": {
        "--transformer_clf_gpu_id": 0,
        "--use_gpu_id": 0,
        "--gpt2_gpu_id": 0,
        "--strategy_gpu_id": 0,
        "--ce_gpu_id": 0,
        "--bert_ppl_gpu_id": 0
    }
}

DATASET_CONFIG = {
    "expert_layman": {
        "--dataset": "expert_layman",
        "--output_dir": "exp-expert_layman",
        "--bert_clf_steps": 20000
    },
    "formality": {
        "--dataset": "GYAFC_Corpus",
        "--output_dir": "exp-formality",
        "--bert_clf_steps": 20000
    },
    "gender": {
        "--dataset": "gender_data",
        "--output_dir": "exp-gender",
        "--bert_clf_steps": 20000
    },
    "politeness": {
        "--dataset": "politeness_data",
        "--output_dir": "exp-politeness",
        "--bert_clf_steps": 20000
    },
    "political": {
        "--dataset": "political_data",
        "--output_dir": "exp-political",
        "--bert_clf_steps": 20000
    },
}

STRATEGY_CONFIG = {
    "identity": {
        "--strategy": "IdentityStrategy"
    },
    "cheat": {
        "--strategy": "CheatStrategy"
    },
    "ssrs": {
        "--strategy": "SSRSv2Strategy",
        "--ssrs_sim_threshold": 0.8,
        "--ssrs_sim_weight": 20,
        "--ssrs_ppl_weight": 3,
        "--ssrs_sampling_steps": 100,
        "--ssrs_clf_weight": 3,
        "--ssrs_window_size": 3,
        "--ssrs_accept_criteria": "joint_weighted_criteria",
        "--ssrs_sim_metric": "USESimilarityMetric",
        "--ssrs_early_stop": "none",
        "--ssrs_bleu_weight": 10,
        "--ssrs_bleu_threshold": 0.6,
    },
}


def to_command(args):
    ret = []
    for k, v in args.items():
        ret.append(k)
        ret.append(str(v))

    return ret


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", choices=list(GPU_CONFIG.keys()), default="single")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    parser.add_argument("--strategy", choices=list(STRATEGY_CONFIG.keys()) + ["all"],
                        default="all")
    parser.add_argument("--offset", type=int, default=0)

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
            command = ["python3", "-m", "fibber.benchmark.benchmark_style_transfer"]
            command += to_command(COMMON_CONFIG)
            command += to_command(GPU_CONFIG[args.gpu])
            command += to_command(DATASET_CONFIG[dataset])
            command += to_command(STRATEGY_CONFIG[strategy])
            command += to_command({"--subsample_offset": args.offset})
            subprocess.call(command)


if __name__ == "__main__":
    main()
