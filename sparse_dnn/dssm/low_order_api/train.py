#coding:utf8
import sys
import os
import numpy as np

import tensorflow as tf

from dssm import DSSM
import data_utils

flags = tf.app.flags
flags.DEFINE_integer('epoch', 1, 'max train steps')
flags.DEFINE_integer('batch_size', 8, 'max train steps')
FLAGS = flags.FLAGS
model_path = "./model/dssm.ckpt"

batch_size = 8

def get_batch_data(step, batch_size, raw_data):
  start = step * batch_size
  end = (step + 1) * batch_size
  return data_utils.load_dataset_batch(raw_data[start:end])

def train():
  dssm = DSSM()
  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.epoch):
      train_raw_data = data_utils.load_all_dataset("0107")
      if train_raw_data is None:
        continue
      epoch_steps = int(len(train_raw_data) / FLAGS.batch_size)
      for step in range(epoch_steps):
        query_batch, doc_batch, label_batch = get_batch_data(step, FLAGS.batch_size, train_raw_data)
        # print(query_batch)
        #print('label:', label_batch)

        print('loss:', sess.run(dssm.loss, feed_dict={dssm.query : query_batch, dssm.doc : doc_batch, dssm.label : label_batch}))
        # print('score:', sess.run(dssm.score, feed_dict={dssm.query : query_batch, dssm.doc : doc_batch}))
        sess.run(dssm.train_step, feed_dict={dssm.query : query_batch, dssm.doc : doc_batch, dssm.label : label_batch})
    saver.save(sess, model_path)

if __name__ == '__main__':
  train()
  sys.exit(0)
