import os

import tensorflow as tf
import tensorflow_hub as hub

from .. import log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')
logger = log.setup_custom_logger('measure-use')


class USESimilarity(object):
  def __init__(self):
    super(USESimilarity, self).__init__()
    logger.info("load universal sentence encoder")
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    self.embed = hub.Module(module_url)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.build_graph()
    self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

  def build_graph(self):
    self.sts_input1 = tf.placeholder(tf.string, shape=(None))
    self.sts_input2 = tf.placeholder(tf.string, shape=(None))

    sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
    sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
    self.cosine_similarities = tf.reduce_sum(
        tf.multiply(sts_encode1, sts_encode2), axis=1)
    self.sim_scores = self.cosine_similarities

  def __call__(self, s1, s2):
    scores = self.sess.run(
        [self.sim_scores],
        feed_dict={
            self.sts_input1: [s1],
            self.sts_input2: [s2],
        })
    return float(scores[0][0])


if __name__ == "__main__":
  measure = USESimilarity()
  s1 = "a is a a a a"
  s2 = "rose is a rose"
  print(s1, s2, measure(s1, s2), sep="\n")

  s1 = "Saturday is the last day in a week"
  s2 = "Sunday is the last day in a week"
  print(s1, s2, measure(s1, s2), sep="\n")
