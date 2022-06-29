import pytest
import torch

from fibber.benchmark import Benchmark
from fibber.paraphrase_strategies import ASRSStrategy, TextAttackStrategy


@pytest.fixture()
def gpu_id():
    if torch.cuda.device_count() > 0:
        return 0
    return -1


@pytest.mark.slow
def test_integrity_identity(gpu_id):
    torch.cuda.empty_cache()
    benchmark = Benchmark(
        output_dir="exp-pytest",
        dataset_name="movie_review",
        subsample_attack_set=100,
        use_gpu_id=gpu_id,
        bert_ppl_gpu_id=gpu_id,
        transformer_clf_gpu_id=gpu_id,
        transformer_clf_steps=5000,
        transformer_clf_bs=32
    )

    result = benchmark.run_benchmark(paraphrase_strategy="IdentityStrategy")
    assert result["bert-base-cased-Classifier_Accuracy_targeted(↓)"] > 0.85


@pytest.mark.slow
def test_integrity_textfooler(gpu_id):
    torch.cuda.empty_cache()
    benchmark = Benchmark(
        output_dir="exp-pytest",
        dataset_name="movie_review",
        subsample_attack_set=100,
        use_gpu_id=gpu_id,
        bert_ppl_gpu_id=gpu_id,
        transformer_clf_gpu_id=gpu_id,
        transformer_clf_steps=5000,
        transformer_clf_bs=32
    )

    strategy = TextAttackStrategy(
        arg_dict={"ta_recipe": "TextFoolerJin2019"},
        dataset_name="movie_review",
        strategy_gpu_id=gpu_id,
        output_dir="exp-pytest",
        metric_bundle=benchmark.get_metric_bundle(),
        field="text0")
    result = benchmark.run_benchmark(paraphrase_strategy=strategy)

    assert result["bert-base-cased-Classifier_Accuracy_targeted(↓)"] < 0.50


@pytest.mark.slow
def test_integrity_asrs(gpu_id):
    torch.cuda.empty_cache()
    benchmark = Benchmark(
        output_dir="exp-pytest",
        dataset_name="movie_review",
        subsample_attack_set=1,
        use_gpu_id=gpu_id,
        bert_ppl_gpu_id=gpu_id,
        transformer_clf_gpu_id=gpu_id,
        transformer_clf_steps=5000,
        transformer_clf_bs=32
    )
    strategy = ASRSStrategy(
        arg_dict={},
        dataset_name="movie_review",
        strategy_gpu_id=gpu_id,
        output_dir="exp-pytest",
        metric_bundle=benchmark.get_metric_bundle(),
        field="text0")
    benchmark.run_benchmark(paraphrase_strategy=strategy, max_paraphrases=10)
