#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

learning_rate = 0.5
train_epoches = 1000
num_batch = 1
# [None, 784]表示不一定feed多少个x样本,但是每个特征维度固定是784
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

W_h1 = tf.Variable(tf.zeros([784, 512]), name='weight_hidden1')
b_h1 = tf.Variable(tf.zeros([512]), name='bais_hidden1')
W_h2 = tf.Variable(tf.random_normal([512, 1024]), name='weight_hidden2')
b_h2 = tf.Variable(tf.random_normal([1024]), name='bais_hidden2')
W_o = tf.Variable(tf.random_normal([1024, 10]), name='weights_output')
b_o = tf.Variable(tf.ranom_normal([10]), name='bais_output')

model = tf.matmul(x, W_h1) + b_h1
y_predict1 = tf.nn.softmax(model)

#must be tf.reduce_mean(- tf.sum(...))
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(y_predict1), reduction_indices=[1]))
#cross_entropy = - tf.reduce_sum(y * tf.log(y_predict), reduction_indices=[1])
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)
#traning model
for i in range(train_epoches):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

#test model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

sess.close()