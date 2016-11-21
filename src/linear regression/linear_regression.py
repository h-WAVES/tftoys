import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def check(w1, b1, w2, b2, threshold):
    if abs(w1 - w2) <= threshold and abs(b1 - b2) < threshold:
        return True
    return False

num_points = 1000
vector_set = []
for i in range(num_points):
    x = np.random.normal(0.0, 0.55)
    y = 0.1*x + 0.3 + np.random.normal(0.0, 0.03)
    vector_set.append([x, y])

x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

#plt.plot(x_data, y_data, 'ro', label='Original data')
#plt.legend()
#plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

sess.run(train)
w1 = sess.run(W)
b1 = sess.run(b)
for step in xrange(100):
    sess.run(train)
    w2 = sess.run(W)
    b2 = sess.run(b)
    print('w2 rank: ', tf.rank(w2), tf.rank(w2).eval(session=sess))
    if check(w1, b1, w2, b2, 0.00001):
        break
    print(step, sess.run(W), sess.run(b), sess.run(loss))
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, w2*x_data + b2)
    plt.legend()
    #plt.show()
    w1 = w2
    b1 = b2


sess.close()

