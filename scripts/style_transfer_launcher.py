import argparse
import copy
import subprocess

COMMON_CONFIG = {
    "--subsample_testset": 0,
    "--num_paraphrases_per_text": 50,
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
}

STRATEGY_CONFIG = {
    "identity": {
        "--strategy": "IdentityStrategy"
    },
    "cheat": {
        "--strategy": "CheatStrategy"
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

    parser.add_argument("--gpu", choices=list(GPU_CONFIG.keys()), default="single")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    parser.add_argument("--strategy", choices=list(STRATEGY_CONFIG.keys()) + ["all"],
                        default="all")

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
            subprocess.call(command)


if __name__ == "__main__":
    main()
