import argparse
import copy
import subprocess

COMMON_CONFIG = {
    "--subsample_testset": 1000,
    "--num_paraphrases_per_text": 50,
    "--robust_tuning": "0",
    # ignored when robut_tuning is 0 and load_robust_tuned_clf is not set
    "--robust_tuning_steps": 3000,
}

DEFENSE_CONFIG = {
    "none": {},
    "sem": {
        "--bert_clf_enable_sem": "1"
    },
    "lmag": {
        "--bert_clf_enable_lmag": "1"
    }
}

GPU_CONFIG = {
    "0": {
        "--bert_gpu_id": 0,
        "--use_gpu_id": 0,
        "--gpt2_gpu_id": 0,
        "--strategy_gpu_id": 0,
        "--ce_gpu_id": 0
    },
    "1": {
        "--bert_gpu_id": 1,
        "--use_gpu_id": 1,
        "--gpt2_gpu_id": 1,
        "--ce_gpu_id": 1,
        "--strategy_gpu_id": 1,
    },
    "2": {
        "--bert_gpu_id": 2,
        "--use_gpu_id": 2,
        "--gpt2_gpu_id": 2,
        "--ce_gpu_id": 2,
        "--strategy_gpu_id": 2,
    },
    "3": {
        "--bert_gpu_id": 3,
        "--use_gpu_id": 3,
        "--gpt2_gpu_id": 3,
        "--ce_gpu_id": 3,
        "--strategy_gpu_id": 3,
    },
    "mix": {
        "--bert_gpu_id": 3,
        "--use_gpu_id": 3,
        "--gpt2_gpu_id": 3,
        "--ce_gpu_id": 3,
        "--strategy_gpu_id": 0,
    }
}

DATASET_CONFIG = {
    "ag": {
        "--dataset": "ag_no_title",
        "--output_dir": "exp-ag",
        "--bert_clf_steps": 20000
    },
    "mr": {
        "--dataset": "mr",
        "--output_dir": "exp-mr",
        "--bert_clf_steps": 5000
    },
    # "imdb": {
    #     "--dataset": "imdb",
    #     "--output_dir": "exp-imdb",
    #     "--bert_clf_steps": 5000
    # },
    # "yelp": {
    #     "--dataset": "yelp",
    #     "--output_dir": "exp-yelp",
    #     "--bert_clf_steps": 20000
    # },
    "snli": {
        "--dataset": "snli",
        "--output_dir": "exp-snli",
        "--bert_clf_steps": 20000
    },
    "sst2": {
        "--dataset": "sst2",
        "--output_dir": "exp-sst2",
        "--bert_clf_steps": 20000
    }
    # "mnli": {
    #     "--dataset": "mnli",
    #     "--output_dir": "exp-mnli",
    #     "--bert_clf_steps": 20000
    # },
    # "mnli_mis": {
    #     "--dataset": "mnli_mis",
    #     "--output_dir": "exp-mnli_mis",
    #     "--bert_clf_steps": 20000
    # },
    # "qnli": {
    #     "--dataset": "qnli",
    #     "--output_dir": "exp-qnli",
    #     "--bert_clf_steps": 20000
    # },
}

STRATEGY_CONFIG = {
    "identity": {
        "--strategy": "IdentityStrategy"
    },
    "textfooler": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "TextFoolerJin2019",
        "--robust_tune_num_attack_per_step": 20
    },
    "pso": {
        "--strategy": "OpenAttackStrategy",
        "--oa_recipe": "PSOAttacker",
        "--robust_tune_num_attack_per_step": 5
    },
    "bertattack": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "BERTAttackLi2020",
        "--robust_tune_num_attack_per_step": 5
    },
    "bertattack-oa": {
        "--strategy": "OpenAttackStrategy",
        "--oa_recipe": "BERTAttacker",
        "--robust_tune_num_attack_per_step": 5
    },
    "bae": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "BAEGarg2019",
        "--robust_tune_num_attack_per_step": 5
    },
    "clare": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "CLARE2020",
        "--robust_tune_num_attack_per_step": 16
    },
    "scpn": {
        "--strategy": "OpenAttackStrategy",
        "--oa_recipe": "SCPNAttacker",
        "--robust_tune_num_attack_per_step": 5
    },
    "gsa": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "Kuleshov2017",
        "--robust_tune_num_attack_per_step": 5
    },
    "pwws": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "PWWSRen2019",
        "--robust_tune_num_attack_per_step": 5
    },
    "asrs": {
        "--strategy": "ASRSStrategy",
        "--asrs_enforcing_dist": "wpe",
        "--asrs_wpe_threshold": 1.0,
        "--asrs_wpe_weight": 1000,
        "--asrs_sim_threshold": 1.0,
        "--asrs_sim_weight": 500,
        "--asrs_ppl_weight": 100,
        "--asrs_sampling_steps": 50,
        "--asrs_burnin_steps": 0,
        "--asrs_clf_weight": 3,
        "--asrs_window_size": 3,
        "--asrs_accept_criteria": "joint_weighted_criteria",
        "--asrs_burnin_enforcing_schedule": "1",
        "--asrs_burnin_criteria_schedule": "1",
        "--asrs_seed_option": "dynamic_len",
        "--asrs_lm_option": "finetune",
        "--asrs_sim_metric": "CESimilarityMetric",
        "--robust_tune_num_attack_per_step": 5
    },
    "asrs-adv-train": {
        "--asrs_sampling_steps": 50,
        "--asrs_burnin_steps": 25,
        "--asrs_sim_metric": "USESimilarityMetric",
        "--robust_tune_num_attack_per_step": 16
    },
    "fu": {
        "--strategy": "FudgeStrategy",
    },
    "asrs-nli": {
        "--asrs_sim_weight": 100,
        "--asrs_ppl_weight": 50,
        "--asrs_clf_weight": 3,
    },
    "asrs-u": {
        "--asrs_sim_metric": "USESimilarityMetric",
        "--best_adv_metric_name": "USESimilarityMetric"
    },
    "asrs-u-nli": {
        "--asrs_sim_weight": 100,
        "--asrs_ppl_weight": 50,
        "--asrs_clf_weight": 3,
        "--asrs_sim_metric": "USESimilarityMetric",
        "--best_adv_metric_name": "USESimilarityMetric"
    },
    "asrsv2": {
        "--strategy": "ASRSv2Strategy",
        "--asrs2_enforcing_dist": "wpe",
        "--asrs2_wpe_threshold": 1.0,
        "--asrs2_wpe_weight": 5,
        "--asrs2_sim_threshold": 0.95,
        "--asrs2_sim_weight": 10,
        "--asrs2_ppl_weight": 5,
        "--asrs2_sampling_steps": 200,
        "--asrs2_clf_weight": 10,
        "--asrs2_window_size": 3,
        "--asrs2_accept_criteria": "joint_weighted_criteria",
        "--asrs2_lm_option": "finetune",
        "--asrs2_sim_metric": "USESimilarityMetric",
        "--asrs2_early_stop": "half",
        "--robust_tune_num_attack_per_step": 16
    },

    "asrsv2A": {
        "--strategy": "ASRSv2Strategy",
        "--asrs2_enforcing_dist": "wpe",
        "--asrs2_wpe_threshold": 1.0,
        "--asrs2_wpe_weight": 5,
        "--asrs2_sim_threshold": 0.90,
        "--asrs2_sim_weight": 20,
        "--asrs2_ppl_weight": 5,
        "--asrs2_sampling_steps": 200,
        "--asrs2_clf_weight": 10,
        "--asrs2_window_size": 3,
        "--asrs2_accept_criteria": "joint_weighted_criteria",
        "--asrs2_lm_option": "finetune",
        "--asrs2_sim_metric": "USESimilarityMetric",
        "--asrs2_early_stop": "half",
        "--robust_tune_num_attack_per_step": 16
    },

    "asrsv2-1": {
        "--strategy": "ASRSv2Strategy",
        "--asrs2_enforcing_dist": "wpe",
        "--asrs2_wpe_threshold": 1.0,
        "--asrs2_wpe_weight": 1,
        "--asrs2_sim_threshold": 0.8,
        "--asrs2_sim_weight": 50,
        "--asrs2_ppl_weight": 5,
        "--asrs2_sampling_steps": 100,
        "--asrs2_clf_weight": 5,
        "--asrs2_window_size": 3,
        "--asrs2_accept_criteria": "joint_weighted_criteria",
        "--asrs2_lm_option": "finetune",
        "--asrs2_sim_metric": "USESimilarityMetric",
        "--asrs2_early_stop": "all",
        "--robust_tune_num_attack_per_step": 32
    },

    "asrsv2-2": {
        "--strategy": "ASRSv2Strategy",
        "--asrs2_enforcing_dist": "wpe",
        "--asrs2_wpe_threshold": 1.0,
        "--asrs2_wpe_weight": 5,
        "--asrs2_sim_threshold": 0.9,
        "--asrs2_sim_weight": 50,
        "--asrs2_ppl_weight": 3,
        "--asrs2_sampling_steps": 100,
        "--asrs2_clf_weight": 10,
        "--asrs2_window_size": 3,
        "--asrs2_accept_criteria": "joint_weighted_criteria",
        "--asrs2_lm_option": "finetune",
        "--asrs2_sim_metric": "USESimilarityMetric",
        "--asrs2_early_stop": "all",
        "--robust_tune_num_attack_per_step": 32
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

    parser.add_argument("--gpu", choices=list(GPU_CONFIG.keys()), default="0")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    parser.add_argument("--strategy", choices=list(STRATEGY_CONFIG.keys()) + ["all"],
                        default="all")
    parser.add_argument("--robust_desc", type=str, default=None)
    parser.add_argument("--robust_tuning", type=str, default="0")
    parser.add_argument("--defense", type=str, default="none")

    args = parser.parse_args()

    if args.robust_tuning == "1":
        COMMON_CONFIG["--robust_tuning"] = "1"

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
            command += to_command(DEFENSE_CONFIG[args.defense])
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
