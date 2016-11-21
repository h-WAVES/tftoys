#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

"""
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
"""

def generate_data(num_points):
    vector_set = []
    for i in range(num_points):
        x = np.random.normal(0.0, 0.55)
        y = 3  - 2 * x + 5 * math.pow(x, 3) + 2 * math.pow(x, 5)
        vector_set.append([x, y])

    x = [v[0] for v in vector_set]
    y = [v[1] for v in vector_set]
    return (x, y)

# train data is too much to overfit, if decrease the train data number,
# it will increase the train steps
# 根据实际训练样本数量和模型复杂度,适当调整训练次数和样本数量
x_train, y_train = generate_data(1000)
x_test, y_test = generate_data(200)

#plt.plot(x_train, y_train)
#print(x_train, y_train)
#plt.show()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#b = tf.Variable(tf.random_normal([1]), name='bais')
#y_predict = b
W = tf.Variable(tf.random_normal([4], name='weights'))

y_predict = tf.mul(x, W[0]) + tf.mul(tf.pow(x, 3), W[1]) + tf.mul(tf.pow(x, 5), W[2]) + W[3]

#L2 regularization
loss = tf.reduce_mean(tf.square(y - y_predict)) + 0.5 * tf.reduce_sum(tf.abs(W))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_model = optimizer.minimize(loss)
train_epoches = 3

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

weights = sess.run(W)
print(weights)

prev_loss = 10000
for i in range(train_epoches):

    for (xx, yy) in zip(x_train, y_train):
        sess.run(train_model, feed_dict={x: xx, y: yy})
    y_plot = sess.run(y_predict, feed_dict={x: x_train})
    #print(y_plot)
    plt.plot(x_train, y_train, 'ro', color='r')
    plt.plot(x_train, y_plot, color='b')
    plt.show()
    training_loss = sess.run(loss, feed_dict={x: x_train, y: y_train})
    if abs(training_loss - prev_loss) < 0.00001:
        break
    weights = sess.run(W)
    print('model : %.3f*x + %.3f*x^2 + %.3f*x^3 + %.3f' % (weights[0], weights[1], weights[2], weights[3]))

    print('training step %d,  model loss is %.6f' %(i+1, training_loss))
    prev_loss = training_loss

#plt.show()
testing_loss = sess.run(loss, feed_dict={x: x_test, y: y_test})
print('test loss : %.6f' % testing_loss)
sess.close()
