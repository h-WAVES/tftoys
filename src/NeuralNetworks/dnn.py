#coding:utf-8
import numpy as np
import timeit
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class DNN():
    def __init__(self):
        self.mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
        self.trX, self.trY, self.teX, self.teY = self.mnist.train.images, self.mnist.train.labels, self.mnist.test.images, self.mnist.test.labels
        self.learning_rate = 0.05
        self.training_epoches = 10
        self.num_batch = 200

        self.pro_keep_input = 0.8
        self.pro_keep_hidden = 0.5

        #too large to overfit
        self.neurals_hidden1 = 512
        self.neurals_hidden2 = 1024

        self.n_input = 784
        self.n_output = 10

    # dynamic layers settings
    def init_parameters(self, n_input, n_output, layers, learning_rate, training_epoches, num_batch, weights, bais):
        self.n_input = n_input
        self.n_output = n_output
        self.layers = layers
        self.weights = weights
        self.bais = bais
        self.learning_rate = learning_rate
        self.training_epoches = training_epoches
        self.num_batch = num_batch
        """
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
        }
        bais = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }
        """

    def init_weights(self, shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def dnn_model(self, x, w_h1, b_h1, w_h2, b_h2, w_o, b_o, pro_keep_input, pro_keep_hidden):
        x = tf.nn.dropout(x, pro_keep_input)
        #print(x, w_h1)
        h1 = tf.nn.relu(tf.add(tf.matmul(x, w_h1), b_h1))
        h1 = tf.nn.dropout(h1, pro_keep_hidden)

        h2 = tf.nn.relu(tf.add(tf.matmul(h1, w_h2), b_h2))
        h2 = tf.nn.dropout(h2, pro_keep_hidden)

        return tf.add(tf.matmul(h2, w_o), b_o)

    def run(self):
        x = tf.placeholder('float', [None, 784])
        y = tf.placeholder('float', [None, 10])

        #input, output = self.shape
        w_h1 = self.init_weights([self.n_input, self.neurals_hidden1], 'weight_hidden1') #784 * 512
        b_h1 = self.init_weights([self.neurals_hidden1], 'bais_hidden1') # 512
        w_h2 = self.init_weights([self.neurals_hidden1, self.neurals_hidden2], 'weight_hidden2') # 512 * 1024
        b_h2 = self.init_weights([self.neurals_hidden2], 'bais_hidden2') # 1024
        w_o = self.init_weights([self.neurals_hidden2, self.n_output], 'weights_output') # 1024 * 10
        b_o = self.init_weights([self.n_output], 'bais_output') # 10

        y_predict = self.dnn_model(x, w_h1, b_h1, w_h2, b_h2, w_o, b_o, self.pro_keep_input, self.pro_keep_hidden)
        # tf.reduce_mean -> batch train
        #softmax_cross_entropy_with_logits (y_predict, unscaled logits)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y))
        train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
        predict_op = tf.argmax(y_predict, 1)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        #Train DNN model
        avg_cost = 0.
        total_batch = int(self.mnist.train.num_examples / self.num_batch)
        for epoch in range(self.training_epoches):
            start = timeit.default_timer()
            for i in range(total_batch):
                batch_x, batch_y = self.mnist.train.next_batch(self.num_batch)
                _, c = sess.run([train_op, cost], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            end = timeit.default_timer()
            print("DNN training step: %04d time cost %.6f seconds" % (epoch + 1, end - start), "loss={:.9f}".format(avg_cost))
        print("Training Deep Neural Network Finished!")

        # Test DNN
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: self.mnist.test.images, y: self.mnist.test.labels}, session=sess))

        sess.close()


def main():
    dnn = DNN()
    dnn.run()

main()
