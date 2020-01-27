# coding:utf8
import sys
import tensorflow as tf
import numpy as np

class DataIterator(object):

  def __init__(self, params):
    self._params = params
    self.record_seperator = '\t'
    self.multi_feature_seperator = ','
  
  def decode_csv(self, value):
     features = {}
     params = self._params
     # input format: label\tfeature1\tfeature2\t...
     # feature number = slot_num
     records = tf.decode_csv(value,
                             record_defaults = [[0.0]] * 1 + [['']] * params['slot_num'], 
                             field_delim = self.record_seperator)
     for i in range(1, len(records)):
       sparse_col_string = tf.string_split([records[i]], delimiter = self.multi_feature_seperator)
       # sparse_col_string = tf.string_to_number(sparse_col_string, tf.int64)
       # sparse_col_int_val = tf.string_to_hash_bucket_fast(sparse_col_string.values, num_buckets = sys.maxsize)
       sparse_tensor = tf.SparseTensor(indices = sparse_col_string.indices,
                                       # values = tf.mod(sparse_col_int_val, params["num_buckets"]),
                                       values = sparse_col_string.values,
                                       dense_shape = sparse_col_string.dense_shape)
       features[str(i)] = sparse_tensor
     labels = tf.reshape(records[0], [-1])
     return features, labels
  

  def input_fn(self, input_data, mode):
    if mode == 'online':
      return self.decode_csv(input_data)
    params = self._params
    # dataset = tf.data.TextLineDataset.list_files(input_data)
    dataset = tf.data.TextLineDataset(input_data)
    dataset = dataset.shuffle(buffer_size=params["shuffle_buffer_size"], seed=123)
    dataset = dataset.map(self.decode_csv, num_parallel_calls=params['num_parallel_calls']) \
                          .repeat(params['repeat_num']) \
                          .batch(params['batch_size'])
    return dataset


if __name__ == '__main__':
  tf.enable_eager_execution()
  file_pattern = "./data/part-00000"
  file_pattern = "./data/part-5"
  params = {"slot_num" : 10,
            "shuffle_buffer_size" : 1000,
            "cycle_length" : 1,
            "block_length" : 16,
            "num_parallel_calls" : 8,
            "repeat_num" : 1,
            "batch_size" : 8,
            "num_buckets" : 500000}
  data_iterator = DataIterator(params)
  iterator = data_iterator.input_fn(file_pattern, 'offline')
  print("offline mode")
  for x, y in iterator:
    # print(y)
    for i in x:
      # print(i, x[i])
      pass
  print("online mode")
  with open(file_pattern, 'r') as infile:
    idx = 0
    for line in infile.readlines():
      line = line.strip('\n')
      x, y = data_iterator.input_fn(line, 'online')
      print(idx)
      idx += 1
      for i in x:
        print(x[i].values, end = '')
        print(" ", end = '')
      print()
