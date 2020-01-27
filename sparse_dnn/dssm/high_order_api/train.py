#coding:utf8
import sys
import os

import numpy as np
import tensorflow as tf

from dssm import DSSM
from data_iterator import DataIterator

model_path = "./model/dssm.ckpt"

train_params = {"shuffle_buffer_size" : 1000,
                "num_parallel_calls" : 4,
                "epoch" : 10,
                "batch_size" : 4}
data_iterator = DataIterator(train_params)

data_file = "./data/train.txt.10"

def train():
  dssm = DSSM()
  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    iterator = data_iterator.input_fn(data_file)

    sess.run(iterator.initializer)
    while True:
      try:
        (query_features, creative_ids, labels) = iterator.get_next()
        (batch_query, batch_creative_ids, batch_labels) = sess.run([query_features, creative_ids, labels])
        # print(sess.run([query_features, creative_ids, labels]))
        # print('loss:', sess.run(dssm.loss, feed_dict={dssm.query : batch_query, dssm.doc : batch_creative_ids, dssm.label : batch_labels}))
        sess.run(dssm.train_step, feed_dict={dssm.query : batch_query, dssm.doc : batch_creative_ids, dssm.label : batch_labels})
        print('score:', sess.run(dssm.score, feed_dict={dssm.query : batch_query, dssm.doc : batch_creative_ids}))
      except tf.errors.OutOfRangeError:
        break
    saver.save(sess, model_path)

if __name__ == '__main__':
  train()
  sys.exit(0)
