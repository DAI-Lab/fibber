import argparse
import copy
import subprocess

COMMON_CONFIG = {
    "--subsample_testset": 1000,
    "--max_paraphrases": 50,
    "--task": "attack",
    "--target_classifier": "transformer",
    "--transformer_clf_model_init": "bert-base"
}

DEFENSE_CONFIG = {
    "none": {},
    "lmag": {
        "--defense_strategy": "LMAgStrategy"
    },
    "sem": {
        "--defense_strategy": "SEMStrategy",
    },
    "adv": {
        "--defense_strategy": "AdvTrainStrategy",
    },
    "sapd": {
        "--defense_strategy": "SAPDStrategy",
    }
}

GPU_CONFIG = {
    "cpu": {
        "--transformer_clf_gpu_id": -1,
        "--bert_ppl_gpu_id": -1,
        "--use_gpu_id": -1,
        "--strategy_gpu_id": -1,
    },
    "gpu": {
        "--transformer_clf_gpu_id": 0,
        "--bert_ppl_gpu_id": 0,
        "--use_gpu_id": 0,
        "--strategy_gpu_id": 0,
    },
}

DATASET_CONFIG = {
    "ag_news": {
        "--dataset": "ag_news",
        "--output_dir": "exp-ag_news",
        "--transformer_clf_steps": 20000
    },
    "movie_review": {
        "--dataset": "movie_review",
        "--output_dir": "exp-movie_review",
        "--transformer_clf_steps": 5000
    },
    "sst2": {
        "--dataset": "sst2",
        "--output_dir": "exp-sst2",
        "--transformer_clf_steps": 20000
    },
    "trec": {
        "--dataset": "trec",
        "--output_dir": "exp-trec",
        "--transformer_clf_steps": 5000
    },
    "twitter_toxicity": {
        "--dataset": "twitter_toxicity",
        "--output_dir": "exp-twitter_toxicity",
        "--transformer_clf_steps": 20000
    },
    "fake_news": {
        "--dataset": "fake_news",
        "--output_dir": "exp-fake_news",
        "--transformer_clf_steps": 20000
    },
    "fake_news_title": {
        "--dataset": "fake_news_title",
        "--output_dir": "exp-fake_news_title",
        "--transformer_clf_steps": 20000
    },
    "fake_review_generated": {
        "--dataset": "fake_review_generated",
        "--output_dir": "exp-fake_review_generated",
        "--transformer_clf_steps": 20000
    },
}

STRATEGY_CONFIG = {
    "identity": {
        "--strategy": "IdentityStrategy"
    },
    "textfooler": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "TextFoolerJin2019",
    },
    "pso": {
        "--strategy": "OpenAttackStrategy",
        "--oa_recipe": "PSOAttacker",
    },
    "bertattack": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "BERTAttackLi2020",
    },
    "bae": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "BAEGarg2019",
    },
    "clare": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "CLARE2020",
    },
    "a2t": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "A2TYoo2021",
    },
    "scpn": {
        "--strategy": "OpenAttackStrategy",
        "--oa_recipe": "SCPNAttacker",
    },
    "gsa": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "Kuleshov2017",
    },
    "pwws": {
        "--strategy": "TextAttackStrategy",
        "--ta_recipe": "PWWSRen2019",
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
        "--asrs_sim_metric": "USESimilarityMetric",
    },
    "fu": {
        "--strategy": "FudgeStrategy",
    },
    "rr": {
        "--strategy": "RewriteRollbackStrategy",
        "--rr_enforcing_dist": "wpe",
        "--rr_wpe_threshold": 1.0,
        "--rr_wpe_weight": 5,
        "--rr_sim_threshold": 0.95,
        "--rr_sim_weight": 20,
        "--rr_ppl_weight": 5,
        "--rr_sampling_steps": 200,
        "--rr_clf_weight": 2,
        "--rr_window_size": 3,
        "--rr_accept_criteria": "joint_weighted_criteria",
        "--rr_lm_option": "finetune",
        "--rr_sim_metric": "USESimilarityMetric",
        "--rr_early_stop": "half",
    },
    "sap": {
        "--strategy": "SapStrategy"
    },
    "rm": {
        "--strategy": "RemoveStrategy"
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

    parser.add_argument("--gpu", choices=list(GPU_CONFIG.keys()), default="gpu")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="movie_review")
    parser.add_argument("--strategy", choices=list(STRATEGY_CONFIG.keys()), default="textfooler")
    parser.add_argument("--task", choices=["attack", "defense"], default="attack")
    parser.add_argument("--defense_strategy", choices=list(DEFENSE_CONFIG.keys()), default="none")
    parser.add_argument("--defense_desc", type=str, default=None)
    parser.add_argument("--subsample_testset", type=int, default=1000)

    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--classifier", type=str, default="bert-base-cased")

    args = parser.parse_args()

    COMMON_CONFIG["--task"] = args.task
    COMMON_CONFIG["--subsample_testset"] = args.subsample_testset

    dataset = args.dataset
    strategy = args.strategy

    command = ["python3", "-m", "fibber.benchmark.benchmark_adversarial_attack"]

    if args.classifier == "fasttext":
        COMMON_CONFIG["--target_classifier"] = "fasttest"
    else:
        COMMON_CONFIG["--transformer_clf_model_init"] = args.classifier

    if args.exp_name is not None:
        command += ["--exp_name", args.exp_name]

    command += to_command(COMMON_CONFIG)
    command += to_command(GPU_CONFIG[args.gpu])
    command += to_command(DATASET_CONFIG[dataset])
    command += to_command(DEFENSE_CONFIG[args.defense_strategy])

    if args.defense_strategy != "none":
        assert args.defense_desc is not None
        command += to_command({"--defense_desc": args.defense_desc})

    command += to_command(STRATEGY_CONFIG[strategy])

    subprocess.call(command)


if __name__ == "__main__":
    main()
