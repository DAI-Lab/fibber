import pytest

from fibber.fibber import Fibber


@pytest.mark.slow
def test_fibber():
    arg_dict = {
        "use_gpu_id": 0,
        "gpt2_gpu_id": 0,
        "bert_gpu_id": 0,
        "strategy_gpu_id": 0
    }
    fibber = Fibber(arg_dict, dataset_name="mr", strategy_name="IdentityStrategy",
                    output_dir="exp-pytest")
    text = "test text."
    paraphrases = fibber.paraphrase({"text0": text})[1]
    for item in paraphrases:
        assert item == text
