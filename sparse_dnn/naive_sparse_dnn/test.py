import sys
import tensorflow as tf
import numpy as np

"""
# test sigmoid_cross_entropy_with_logits
input_data = tf.Variable(np.random.rand(1, 3), dtype=tf.float32)
output = tf.nn.sigmoid_cross_entropy_with_logits(logits=input_data, labels=[[1.0, 0.0, 0.]])
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(output))
print(input_data)
print(sess.run(input_data))
a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=input_data, labels=[[1.0, 0.0, 0.]]))
print(sess.run(a))
"""

# test dataset
tf.enable_eager_execution()
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
file_pattern = "./data/part-5"
# file_pattern = "./data/part-00000"
dataset = tf.data.TextLineDataset.list_files(file_pattern)
dataset = dataset.interleave(lambda filename: (tf.data.TextLineDataset(filename)),
                                               cycle_length=4,
                                               block_length=4,
                                               num_parallel_calls=4)
print(dataset)
# slot_num = 10
def decode_csv(value):
   features = {}
   records = tf.decode_csv(value,
                           record_defaults = [[0.0]] * 1 + [['']] * 10, 
                           field_delim = '\t')
   for i in range(1, len(records)):
     sparse_col_string = tf.string_split([records[i]], delimiter = ',')
     # sparse_col_string = tf.string_to_number(sparse_col_string, tf.int64)
     sparse_col_int_val = tf.string_to_hash_bucket_fast(sparse_col_string.values, num_buckets = sys.maxsize)
     sparse_tensor = tf.SparseTensor(indices = sparse_col_string.indices,
                                     values = sparse_col_int_val,
                                     # values = sparse_col_string.values,
                                     dense_shape = sparse_col_string.dense_shape)
     features[i] = sparse_tensor
   # labels = tf.reshape(records[0], [1, 1])
   labels = tf.reshape(records[0], [-1])
   return features, labels

# dataset = dataset.map(lambda x: (tf.string_split([x], delimiter="\t").values), num_parallel_calls=4)
dataset = dataset.map(decode_csv, num_parallel_calls=4).repeat(1).batch(4)
for x, y in dataset:
  print(y)
  for i in x:
    print(i, x[i] )
