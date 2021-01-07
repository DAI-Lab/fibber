import pytest
import torch

from fibber.metrics.metric_utils import MetricBundle


@pytest.fixture()
def gpu_id():
    if torch.cuda.device_count() > 0:
        return 0
    return -1


@pytest.mark.slow
def test_metric_bundle(gpu_id):
    metric_bundle = MetricBundle(
        use_gpu_id=gpu_id,
        gpt2_gpu_id=gpu_id)
    s1 = "Saturday is the last day in a week"
    s2 = "Sunday is the last day in a week"
    results = metric_bundle.measure_example(s1, s2)
    assert len(results) == 4
