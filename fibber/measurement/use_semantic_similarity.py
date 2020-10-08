import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .. import log
from .measurement_base import MeasurementBase

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
logger = log.setup_custom_logger(__name__)
logger.root.handlers = []    # tensorflow_hub mess up the python logging


def config_tf_gpu(gpu_id):
    if tf.__version__ >= "2.3.0":
        gpus = tf.config.list_physical_devices(device_type="GPU")
        gpus = [item for item in gpus if item.name.endswith("GPU:%d" % gpu_id)]
        tf.config.set_visible_devices(gpus, device_type="GPU")
        for device in gpus:
            tf.config.set_memory_growth(device, True)
    else:
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        gpus = [item for item in gpus if item.name.endswith("GPU:%d" % gpu_id)]
        tf.config.experimental.set_visible_devices(gpus, device_type="GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)


class USESemanticSimilarity(MeasurementBase):
    def __init__(self, use_gpu_id=-1, **kargs):
        super(USESemanticSimilarity, self).__init__()
        logger.info("load universal sentence encoder")
        config_tf_gpu(use_gpu_id)
        if use_gpu_id == -1:
            logger.warning("Universal sentence encoder is using CPU.")
        else:
            logger.info("Universal sentence encoder measurement is using GPU %d.", use_gpu_id)
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(module_url)
        logger.root.handlers = []    # tensorflow_hub mess up the python logging

    def __call__(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
        origin = " ".join(origin.split()[:200])
        paraphrase = " ".join(paraphrase.split()[:200])
        embs = self.model([origin, paraphrase]).numpy()
        return float(np.sum(embs[0] * embs[1]) / np.linalg.norm(embs[0]) / np.linalg.norm(embs[1]))
