#coding:utf-8
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

learning_rate = 0.5
train_epoches = 1000
num_batch = 1
# [None, 784]表示不一定feed多少个x样本,但是每个特征维度固定是784
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

W = tf.Variable(tf.zeros([784, 10]), name='weight')
b = tf.Variable(tf.zeros([10]), name='bais')

model = tf.matmul(x, W) + b
y_predict = tf.nn.softmax(model)

#must be tf.reduce_mean(- tf.sum(...))
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(y_predict), reduction_indices=[1]))
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