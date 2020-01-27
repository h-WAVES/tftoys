#coding:utf8
import sys
import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('query_length', 165, '')
flags.DEFINE_integer('doc_length', 1, '')
flags.DEFINE_integer('vocab_size', 30000000, '')
flags.DEFINE_integer('embedding_dim', 2, '')
flags.DEFINE_integer('query_layer_1_units', 16, '')
flags.DEFINE_integer('doc_layer_1_units', 16, '')
flags.DEFINE_float('learning_rate', 0.001, '')


class DSSM(object):

  def __init__(self):
    self._build_model()

  def _build_model(self):
    with tf.name_scope('input'):
      # shape : [batch_size, query_length]
      self.query = tf.placeholder(tf.int32, shape=[None, FLAGS.query_length], name="query") 
      # shape: [batch_size, doc_length]
      self.doc = tf.placeholder(tf.int32, shape=[None, FLAGS.doc_length], name="doc")
      # shape : [batch_size, 1]
      self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

    with tf.name_scope('embedding'):
      # shape : [vocab_size, embedding_dim]
      self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]), dtype=tf.float32, name="embedding")

      # shape: [batch_size, query_length, embedding_dim]
      self.query_embeddings = tf.nn.embedding_lookup(self.embedding, self.query, name="query_embeddings")
      self.doc_embeddings = tf.nn.embedding_lookup(self.embedding, self.doc, name="doc_embeddings")

    with tf.name_scope('mask'):
      # mask query input
      self.query_mask = tf.cast(tf.greater(self.query, 0), tf.float32)
      self.query_mask = tf.expand_dims(self.query_mask, axis=2)
      self.query_mask = tf.tile(self.query_mask, (1, 1, FLAGS.embedding_dim))
      self.query_embeddings_mask = tf.multiply(self.query_embeddings, self.query_mask)
      # self.query_embeddings = self.query_embeddings_mask

      self.doc_mask = tf.cast(tf.greater(self.doc, 0), tf.float32)
      self.doc_mask = tf.expand_dims(self.doc_mask, axis=2)
      self.doc_mask = tf.tile(self.doc_mask, (1, 1, FLAGS.embedding_dim))
      self.doc_embeddings_mask = tf.multiply(self.doc_embeddings, self.doc_mask)

    with tf.name_scope('flatten'):
      # flatten tensor after embedding
      # self.query_flatten = tf.reshape(self.query_embeddings, [-1, FLAGS.query_length * FLAGS.embedding_dim])
      # self.doc_flatten = tf.reshape(self.doc_embeddings, [-1, FLAGS.doc_length * FLAGS.embedding_dim])
      # flatten tensor after embedding and mask empty or default features
      self.query_flatten = tf.reshape(self.query_embeddings_mask, [-1, FLAGS.query_length * FLAGS.embedding_dim])
      self.doc_flatten = tf.reshape(self.doc_embeddings_mask, [-1, FLAGS.doc_length * FLAGS.embedding_dim])

    with tf.name_scope('dense_layer_1'):
      # query_flatten * query_w1
      self.query_w1 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.query_length * FLAGS.embedding_dim, FLAGS.query_layer_1_units)))
      self.query_layer_1 = tf.matmul(self.query_flatten, self.query_w1)

      self.doc_w1 = tf.Variable(tf.glorot_uniform_initializer()((FLAGS.doc_length * FLAGS.embedding_dim, FLAGS.doc_layer_1_units)))
      self.doc_layer_1 = tf.matmul(self.doc_flatten, self.doc_w1)

      self.query_layer_1_out = tf.nn.relu(self.query_layer_1)
      self.doc_layer_1_out = tf.nn.relu(self.doc_layer_1)
    
    with tf.name_scope('cosine_similarity'):
      self.cosine_similarity = tf.reduce_sum(tf.multiply(self.query_layer_1_out, self.doc_layer_1_out), axis=1, keepdims=True)

    with tf.name_scope('score'):
      self.score = tf.nn.sigmoid(self.cosine_similarity)

    with tf.name_scope('loss'):
      self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.label)

    with tf.name_scope('train'):
      self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
      self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
      # SGD

  def load_model(self, saver, sess, model_path):
    if os.path.exists(model_path + '.index'):
      saver.restore(sess, model_path)
    else:
      raise Exception("model_path %s not exist" % model_path+'.index')
      exit(-1)

  def save_model(self, saver, sess, model_path):
    saver.save(sess, model_path)

if __name__ == '__main__':
  dssm = DSSM()
  sys.exit(0)
