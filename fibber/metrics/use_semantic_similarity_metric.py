"""This metric computes the cosine similarity between two sentences. The sentence embedding is
the universal sentence encoder."""
import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from fibber import log, resources
from fibber.metrics.metric_base import MetricBase

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)    # tensorflow_hub mess up the python logging


def config_tf_gpu(gpu_id):
    """Configure tensorflow to use a specific GPU.

    Args:
        gpu_id (int): the gpu id. Set -1 to use CPU.
    """
    if tf.__version__ >= "2.3.0":
        gpus = tf.config.list_physical_devices(device_type="GPU")
        gpus = [item for item in gpus if item.name.endswith("GPU:%d" % gpu_id)]
        tf.config.set_visible_devices(gpus, device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        gpus = [item for item in gpus if item.name.endswith("GPU:%d" % gpu_id)]
        tf.config.experimental.set_visible_devices(gpus, device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)


class USESemanticSimilarityMetric(MetricBase):
    """This metric uses universal sentence encoder to measure the semantic similarity of
    two sentences."""

    def __init__(self, use_gpu_id=-1, **kargs):
        """Initialize universal sentence encoder."""
        super(USESemanticSimilarityMetric, self).__init__()
        logger.info("load universal sentence encoder")
        config_tf_gpu(use_gpu_id)
        if use_gpu_id == -1:
            logger.warning("Universal sentence encoder is using CPU.")
        else:
            logger.info("Universal sentence encoder metric is using GPU %d.", use_gpu_id)
        self.model = hub.load(resources.get_universal_sentence_encoder())
        log.remove_logger_tf_handler(logger)   # tensorflow_hub mess up the python logging

    def measure_batch(self, origin, paraphrase_list, data_record=None, paraphrase_field="text0"):
        """Measure the metric on a batch of paraphrase_list.

        Args:
            origin (str): the original text.
            paraphrase_list (list): a set of paraphrase_list.
            data_record (dict): the corresponding data record of original text.
            paraphrase_field (str): the field name to paraphrase.

        Returns:
            (list): a list containing the USE similarity metric for each paraphrase.
        """
        origin = " ".join(origin.split()[:200])
        paraphrase_list = [" ".join(x.split()[:200]) for x in paraphrase_list]
        embs = self.model([origin] + paraphrase_list).numpy()

        norm = np.linalg.norm(embs, axis=1)
        sim = np.sum(embs[0] * embs, axis=1) / norm[0] / norm
        assert abs(sim[0] - 1) < 1e-4
        return [float(x) for x in sim[1:]]

    def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        """Compute the cosine similarity between the embedding of original text and paraphrased
        text.

        Args:
            origin (str): original text.
            paraphrase (str): paraphrased text.
            data_record: ignored.
            paraphrase_field: ignored.
        """
        origin = " ".join(origin.split()[:200])
        paraphrase = " ".join(paraphrase.split()[:200])
        embs = self.model([origin, paraphrase]).numpy()
        return float(np.sum(embs[0] * embs[1]) / np.linalg.norm(embs[0]) / np.linalg.norm(embs[1]))
