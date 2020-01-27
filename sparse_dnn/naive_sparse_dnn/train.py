# -*- coding: utf-8 -*-
import sys
import os
import itertools
import numpy as np
import tensorflow as tf
from data_iterator import DataIterator
from dnn import SparseDNN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

params = {"slot_num" : 10, 
          "shuffle_buffer_size" : 1000,
          "cycle_length" : 1,
          "block_length" : 1,
          "num_parallel_calls" : 1,
          "repeat_num" : 1,
          "batch_size" : 8,
          "learning_rate" : 0.01,
          "hidden_units" : [16, 8, 4],
          "num_buckets" : 5000000,
          "slot_num" : 10,
          "dimension" : 8}

def model_fn(features, labels, mode, params):
  sparse_dnn = SparseDNN()
  logits = sparse_dnn.dnn_logits(features, labels, mode, params)
  preds = tf.sigmoid(logits)
  # tf.print(preds)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {"probabilities" : preds, "logits" : logits}
    # hook = tf.train.LoggingTensorHook({"preds:" : preds, "logits:" : logits, "input_layer:": input_layer}, every_n_iter=10)
    hook = tf.train.LoggingTensorHook({"preds:" : preds, "logits:" : logits}, every_n_iter=10)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, training_hooks=[hook])
  else:
    predictions = None
  if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, preds)}
  else:
    loss = None
    eval_metric_ops = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  else:
    train_op = None

  # hook = tf.train.LoggingTensorHook({"preds:" : preds, "logits:" : logits, "input_layer:": input_layer}, every_n_iter=10)
  # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, predictions=predictions, training_hooks=[hook])
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, predictions=predictions)

def train(estimator):
  tf.logging.set_verbosity(tf.logging.INFO)
  train_file_pattern = "./data/part-00000"
  data_iterator = DataIterator(params)
  train_input_fn = lambda: data_iterator.input_fn(train_file_pattern, 'offline')
  estimator.train(input_fn=train_input_fn, steps=None)

def eval(estimator):
  eval_file_pattern = "./data/part-5"
  data_iterator = DataIterator(params)
  eval_input_fn = lambda: data_iterator.input_fn(eval_file_pattern, 'offline')
  eval_results = estimator.evaluate(input_fn=eval_input_fn)
  auc_score = eval_results["auc"]
  # print(type(auc_score)) # numpy.float32
  print("\nTest auc: %.6f" % auc_score)
  
def train_and_eval(estimator):
  tf.logging.set_verbosity(tf.logging.INFO)
  # train_file_pattern and eval_file_pattern could be the parameters of FLAGS
  train_file_pattern = "./data/part-00000"
  eval_file_pattern = "./data/part-5"
  data_iterator = DataIterator(params)
  train_input_fn = lambda: data_iterator.input_fn(train_file_pattern, 'offline')
  eval_input_fn = lambda: data_iterator.input_fn(eval_file_pattern, 'offline')
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, start_delay_secs=60, throttle_secs=30)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def predict(estimator):
  predict_file = "./data/part-predict"
  data_iterator = DataIterator(params)
  with open(predict_file, 'r') as infile:
    for line in infile:
      line = line.strip('\n')
      items = line.split('\t')
      dmp_id = items[0]
      ins = "\t".join(items[1:])
      predict_input_fn = lambda: data_iterator.input_fn(ins, 'online')
      predictions = estimator.predict(input_fn=predict_input_fn)   
      predictions = itertools.islice(predictions, 1)
      for i, p in enumerate(predictions):
        print("dmp_id %s: logits:%.6f probability:%.6f" % (dmp_id, p["logits"], p["probabilities"]))

def main():
  train_file_pattern = "./data/part-00000"
  eval_file_pattern = "./data/part-5"
  data_iterator = DataIterator(params)
  train_input_fn = lambda: data_iterator.input_fn(train_file_pattern, 'offline')
  eval_input_fn = lambda: data_iterator.input_fn(eval_file_pattern, 'offline')
  predict_input_fn = lambda: data_iterator.input_fn(eval_file_pattern, 'offline')
  # define estimator
  # estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir="./model")
  estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir="./model")

  # train(estimator)
  # eval(estimator)
  train_and_eval(estimator)
  # predict(estimator)

if __name__ == "__main__":
  main()
  sys.exit(0)
