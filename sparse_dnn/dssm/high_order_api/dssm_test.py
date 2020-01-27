#coding:utf8
import sys
import os
import numpy as np

import tensorflow as tf

from dssm import DSSM

def fake_train_data():
  query = np.random.randint(5, size=[100, 2]) 
  doc = np.random.randint(5, size=[100, 2]) 
  Y = np.random.randint(2, size=[100, 1])
  Y = Y.astype(float)
  return query, doc, Y

def debug():
  query, doc, Y = fake_train_data()
  dssm = DSSM()
  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(len(Y)):
      q = query[i:i+1]
      d = doc[i:i+1]
      label = Y[i:i+1]
      print('query:', sess.run(dssm.query, feed_dict={dssm.query : q}))
      print('doc:', sess.run(dssm.doc, feed_dict={dssm.doc : d}))
      print('label:', sess.run(dssm.label, feed_dict={dssm.label : label}))

      # embedding table
      print('embedding:', sess.run(dssm.embedding))

      # debug query
      print('query_embedding:', sess.run(dssm.query_embeddings, feed_dict={dssm.query : q}))
      print('query_flatten:', sess.run(dssm.query_flatten, feed_dict={dssm.query: q}))
      
      # debug doc
      print('doc_embedding:', sess.run(dssm.doc, feed_dict={dssm.doc : d}))
      print('doc_flatten:', sess.run(dssm.doc_flatten, feed_dict={dssm.doc : d}))

      # debug dense layer
      print('query_layer_1_out:', sess.run(dssm.query_layer_1_out, feed_dict={dssm.query : q}))
      print('doc_layer_1_out:', sess.run(dssm.doc_layer_1_out, feed_dict={dssm.doc : d}))
     
      # debug cosine_similarity, score, loss
      print('cosine_similarity:', sess.run(dssm.cosine_similarity, feed_dict={dssm.query : q, dssm.doc : d}))
      print('score:', sess.run(dssm.score, feed_dict={dssm.query : q, dssm.doc : d}))
      print('loss:', sess.run(dssm.loss, feed_dict={dssm.query : q, dssm.doc : d, dssm.label : label}))


if __name__ == '__main__':
  debug()
  sys.exit(0)
