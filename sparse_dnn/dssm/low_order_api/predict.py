#coding:utf8
import sys
import os
import time

import tensorflow as tf

from dssm import DSSM
import data_utils

# saver = tf.train.Saver()

batch_size = 100

# config
sess = tf.Session()

dssm = DSSM()

meta_path = "./model/dssm.ckpt.meta"
ckpt_path= "./model/dssm.ckpt"

saver = tf.train.import_meta_graph(meta_path)
saver.restore(sess, ckpt_path)

# graph = tf.get_default_graph()

sess.run(tf.global_variables_initializer())

def get_batch_data(step, batch_size, raw_data):
  start = step * batch_size
  end = (step + 1) * batch_size
  return data_utils.load_dataset_batch(raw_data[start:end])

train_raw_data = data_utils.load_all_dataset("0107")
epoch_steps = int(len(train_raw_data) / batch_size)

s = time.time()
for step in range(epoch_steps):
  query_batch, doc_batch, label_batch = get_batch_data(step, batch_size, train_raw_data)
  prediction = sess.run(dssm.score, feed_dict={dssm.query : query_batch, dssm.doc : doc_batch})
  print(prediction)
e = time.time()
# 平均每条 0.0001s
# print(e-s)
