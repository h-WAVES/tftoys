# -*- coding: UTF-8 -*-
import tensorflow as tf

class SparseDNN():
  
  def __init__(self):
    pass

  def get_initializer(self, mode, name):
    if mode != tf.estimator.ModeKeys.TRAIN:
      return tf.zeros_initializer()
    if name == 'he':
      return tf.keras.initializer.he_uniform()
    elif name == 'truncated_normal':
      return tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01)
    elif name == 'glorot':
      return tf.glorot_uniform_initializer()
    raise ValueError("initializer name %s not exists" % name)

  def build_deep_layers(self, net, mode, params):
    for num_hidden_units in params['hidden_units']:
      net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                            kernel_initializer=self.get_initializer(mode, 'glorot'))
    return net

  def get_feature_columns(self, mode, params):
    # initializer = tf.zeros_initializer()
    # initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01)
    initializer = self.get_initializer(mode, "truncated_normal")
    column_list = []
    for i in range(params['slot_num']):
      name = str(i + 1)
      # categorical_column = tf.feature_column.categorical_column_with_identity(key=name, num_buckets=params['num_buckets'], default_value=0)
      # hash_function ?
      categorical_column = tf.feature_column.categorical_column_with_hash_bucket(key=name, hash_bucket_size=params['num_buckets'])
      column_list.append(tf.feature_column.embedding_column(categorical_column, dimension = params['dimension'],
                                                            combiner = 'sum', initializer = initializer))
    return column_list

  def dnn_logits(self, features, labels, mode, params):
    feature_columns = self.get_feature_columns(mode, params)
    input_layer = tf.feature_column.input_layer(features, feature_columns=feature_columns)
    last_deep_layer = self.build_deep_layers(input_layer, mode, params)
    logits = tf.layers.dense(last_deep_layer, units=1,
                             kernel_initializer=self.get_initializer(mode, 'truncated_normal'))
  
    return logits
