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
