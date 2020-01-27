import tensorflow as tf
from data_iterator import DataIterator
# import tensorflow.contrib.eager as tfe

# tf.enable_eager_execution()

params = {"shuffle_buffer_size" : 1000,  "num_parallel_calls" : 4, "epoch" : 10,  "batch_size" : 4}
data_iterator = DataIterator(params)

data_file = "./data/train.txt.10"

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# (features, creative_ids, labels) = data_iterator.input_fn(data_file)
iterator = data_iterator.input_fn(data_file)

sess.run(iterator.initializer)
while True:
  try:
    (query_features, creative_ids, labels) = iterator.get_next()
    print(sess.run([query_features, creative_ids, labels]))
  except tf.errors.OutOfRangeError:
    break
