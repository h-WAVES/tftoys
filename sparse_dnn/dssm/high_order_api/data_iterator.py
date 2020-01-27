#coding:utf8
import tensorflow as tf
from flags import *

# 参考这篇博客 如何用 SparseFeature 结合 Dataset 处理离散特征的
# https://blog.csdn.net/yujianmin1990/article/details/80384994
# dataset + estimator
# https://cloud.tencent.com/developer/article/1063010

# instance format:                                                 
class DataIterator(object):

  def __init__(self, params):
    self._params = params

  # format: _ dmpid creative_id timestamp query_features label
  def decode_csv(self, record):
    # assert length
    records = tf.string_split([record], "\01")
    labels = tf.string_to_number(records.values[-1], tf.float32)
    creative_id = tf.string_to_number(records.values[2], tf.int32)
    query_features = tf.string_split(records.values[4:5], ",")

    query_features = tf.string_to_number(query_features.values, tf.int32)
    query_features = tf.reshape(query_features, [FLAGS.query_length])
    creative_id = tf.reshape(creative_id, [FLAGS.doc_length])
    labels = tf.reshape(labels, [-1])
    return query_features, creative_id, labels

  def input_fn(self, input_file):
    params = self._params
    dataset = tf.data.TextLineDataset(input_file)
    # dataset = dataset.shuffle(buffer_size=params["shuffle_buffer_size"], seed=123)
    dataset = dataset.map(self.decode_csv, num_parallel_calls=params['num_parallel_calls']) \
                     .repeat(params["epoch"]) \
                     .batch(params["batch_size"])
    iterator = dataset.make_initializable_iterator()
    return iterator
