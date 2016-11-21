import tensorflow as tf


#Variable must be initialized first
#because we have specified variables, we must initialize them with initialize_all_variables()
def error_init():
    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    sess = tf.Session()
    print(sess.run(w))
    sess.close()

def right_init():
    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(w))
    sess.close()

right_init()