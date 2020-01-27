#coding:utf8
import sys
import os
import time

import tensorflow as tf

from dssm import DSSM
from data_iterator import DataIterator

def predict(data_params):
  meta_path = "./model/dssm.ckpt.meta"
  ckpt_path= "./model/dssm.ckpt"
  data_file = "./data/train.txt.10"
  dssm = DSSM()
  data_iterator = DataIterator(data_params)
  iterator = data_iterator.input_fn(data_file)
  # config
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, ckpt_path)
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    s = time.time()
    while True:
      try:
        (query_features, creative_ids, labels) = iterator.get_next()
        (batch_query, batch_creative_ids, batch_labels) = sess.run([query_features, creative_ids, labels])
        prediction = sess.run(dssm.score, feed_dict={dssm.query : batch_query, dssm.doc : batch_creative_ids})
        print(prediction)
      except tf.errors.OutOfRangeError:
        break
    e = time.time()
    # 平均每条 0.0001s
    print(e-s)

if __name__ == '__main__':
  data_params = {"shuffle_buffer_size" : 1000,
                 "num_parallel_calls" : 4,
                 "epoch" : 10,
                 "batch_size" : 4}

  # predict
  predict(data_params)

  sys.exit(0)
