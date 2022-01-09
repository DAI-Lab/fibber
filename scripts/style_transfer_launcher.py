import argparse
import copy
import subprocess

COMMON_CONFIG = {
    "--subsample_testset": 500,
    "--num_paraphrases_per_text": 20,
}

GPU_CONFIG = {
    "single": {
        "--bert_gpu_id": 0,
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
    # "cheat": {
    #     "--strategy": "CheatStrategy"
    # },
    "ssrs": {
        "--strategy": "SSRSStrategy",
        "--ssrs_sampling_steps": 100,
        "--ssrs_burnin_steps": 50,
        "--ssrs_window_size": 3,
        "--ssrs_sim_threshold": 0.7,
        "--ssrs_sim_weight": 500,
        "--ssrs_ppl_weight": 1,
        "--ssrs_clf_weight": 10,
        "--ssrs_bleu_weight": 50,
        "--ssrs_bleu_threshold": 0.6,
        "--ssrs_sim_metric": "USESimilarityMetric",
    },
    "ssrsv2": {
        "--strategy": "SSRSv2Strategy",
        "--ssrs2_sim_threshold": 0.8,
        "--ssrs2_sim_weight": 20,
        "--ssrs2_ppl_weight": 3,
        "--ssrs2_sampling_steps": 100,
        "--ssrs2_clf_weight": 3,
        "--ssrs2_window_size": 3,
        "--ssrs2_accept_criteria": "joint_weighted_criteria",
        "--ssrs2_sim_metric": "USESimilarityMetric",
        "--ssrs2_early_stop": "none",
        "--ssrs_bleu_weight": 10,
        "--ssrs_bleu_threshold": 0.6,
    },
    "ssrsv2B": {
        "--strategy": "SSRSv2Strategy",
        "--ssrs2_sim_threshold": 1.0,
        "--ssrs2_sim_weight": 20,
        "--ssrs2_ppl_weight": 3,
        "--ssrs2_sampling_steps": 100,
        "--ssrs2_clf_weight": 3,
        "--ssrs2_window_size": 3,
        "--ssrs2_accept_criteria": "joint_weighted_criteria",
        "--ssrs2_sim_metric": "USESimilarityMetric",
        "--ssrs2_early_stop": "none",
        "--ssrs_bleu_weight": 10,
        "--ssrs_bleu_threshold": 0.6,
    },
    "ssrsv2C": {
        "--strategy": "SSRSv2Strategy",
        "--ssrs2_sim_threshold": 1.0,
        "--ssrs2_sim_weight": 10,
        "--ssrs2_ppl_weight": 3,
        "--ssrs2_sampling_steps": 100,
        "--ssrs2_clf_weight": 3,
        "--ssrs2_window_size": 3,
        "--ssrs2_accept_criteria": "joint_weighted_criteria",
        "--ssrs2_sim_metric": "USESimilarityMetric",
        "--ssrs2_early_stop": "none",
        "--ssrs_bleu_weight": 10,
        "--ssrs_bleu_threshold": 0.6,
    },
    "ssrsv2D": {
        "--strategy": "SSRSv2Strategy",
        "--ssrs2_sim_threshold": 1.0,
        "--ssrs2_sim_weight": 10,
        "--ssrs2_ppl_weight": 3,
        "--ssrs2_sampling_steps": 100,
        "--ssrs2_clf_weight": 3,
        "--ssrs2_window_size": 3,
        "--ssrs2_accept_criteria": "joint_weighted_criteria",
        "--ssrs2_sim_metric": "USESimilarityMetric",
        "--ssrs2_early_stop": "none",
        "--ssrs_bleu_weight": 10,
        "--ssrs_bleu_threshold": 1.0,
    },
    "ssrsv2E": {
        "--strategy": "SSRSv2Strategy",
        "--ssrs2_sim_threshold": 1.0,
        "--ssrs2_sim_weight": 10,
        "--ssrs2_ppl_weight": 10,
        "--ssrs2_sampling_steps": 100,
        "--ssrs2_clf_weight": 3,
        "--ssrs2_window_size": 3,
        "--ssrs2_accept_criteria": "joint_weighted_criteria",
        "--ssrs2_sim_metric": "USESimilarityMetric",
        "--ssrs2_early_stop": "none",
        "--ssrs_bleu_weight": 10,
        "--ssrs_bleu_threshold": 1.0,
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
            if strategy[:4] == "asrs":
                strategy_config_tmp = copy.copy(STRATEGY_CONFIG["asrs"])
                if strategy != "asrs":
                    for k, v in STRATEGY_CONFIG[strategy].items():
                        strategy_config_tmp[k] = v
                command += to_command(strategy_config_tmp)
            else:
                command += to_command(STRATEGY_CONFIG[strategy])
            command += to_command({"--subsample_offset": args.offset})
            subprocess.call(command)


if __name__ == "__main__":
    main()
