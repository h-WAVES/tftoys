#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import timeit
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from tensorflow.examples.tutorials.mnist import input_data


class RNN():
    def __init__(self, learning_rate=0.05,
                 training_epoches=5,
                 batch_size=128,
                 n_input=28,
                 n_steps=28,
                 n_hidden=32,
                 n_classes=10,
                 forget_bias=1.0):
        self.learning_rate = 0.02
        self.training_epoches = 3
        self.batch_size = 128

        self.n_input = 28
        self.n_steps = 28
        self.n_hidden = 32
        self.n_classes = 10
        self.forget_bias = 1.0

        #self.weight = tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        self.weight = tf.Variable(tf.random_uniform([self.n_hidden, self.n_classes]))
        self.bias = tf.Variable(tf.random_normal([self.n_classes]))
        #self.bias = tf.Variable(tf.random_uniform([self.n_classes]))
        self.init_weight = self.weight
        self.init_bias = self.bias
        self.weights = {
            'out' : self.init_weight
            #'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes])),
        }
        self.biases = {
            'out' : self.init_bias
            #'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.mnist = input_data.read_data_sets("data/MNIST_data", one_hot=True)


    def init_weights(self, shape, name=None):
        self.weight = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(self.weight, name=name)
        #return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def rnn_model(self, x, weights, biases, forget_bias=1.0):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.n_input])
        x = tf.split(0, self.n_steps, x)

        lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=forget_bias, state_is_tuple=True)

        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights['out'] + biases['out'])

    def run(self):
        x = tf.placeholder('float', [None, self.n_steps, self.n_input])
        y = tf.placeholder('float', [None, self.n_classes])

        y_predict = self.rnn_model(x, self.weights, self.biases, self.forget_bias)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(cost)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        print('init weight', sess.run(self.weight))
        print('init bias', sess.run(self.bias))
        avg_cost = 0.
        total_batch_num = int(self.mnist.train.num_examples / self.batch_size)
        print(total_batch_num)
        for epoch in range(self.training_epoches):
            start = timeit.default_timer()
            for i in range(total_batch_num):
                batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
                batch_x = batch_x.reshape((self.batch_size, self.n_steps, self.n_input))
                _, c = sess.run([train_op, cost], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c/total_batch_num
            end = timeit.default_timer()
            print("RNN training step: %04d time cost %.6f seconds" % (epoch + 1, end - start), "loss={:.9f}".format(avg_cost))

        # Test RNN
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        x_test_len = self.batch_size
        x_test = self.mnist.test.images[:x_test_len]
        x_test_reshape = x_test.reshape((-1, self.n_steps, self.n_input))
        print("Accuracy:", accuracy.eval({x: x_test_reshape, y: self.mnist.test.labels[:x_test_len]}, session=sess))
        sess.close()

def main():
    rnn = RNN()
    rnn.run()

main()