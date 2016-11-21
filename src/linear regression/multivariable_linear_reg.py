import tensorflow as tf
import numpy as np

learning_rate = 0.05
training_epochs = 10000

x = tf.placeholder("float32", 3)
y = tf.placeholder("float32")

#w = tf.Variable([2.0, 0.5, 3.0], name="w")
w = tf.Variable(tf.random_uniform([3], -1.0, 1.0), name='weight')
#b = tf.Variable(0.1, name='b')
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='bias')
# model is y = w0*x0 + w1*x1 + b
y_model = tf.mul(x, w) + b

# Our error is defined as the square of the differences
loss = tf.square(y - y_model)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.initialize_all_variables()

errors = []
with tf.Session() as session:
    session.run(model)
    print(session.run(w), session.run(b))

    for i in range(training_epochs):
        x_value = np.random.normal(0, 0.1, 3)
        y_value = x_value[0] * 3  + x_value[1] * 1 + x_value[2] * 4 + 6
        _, error_value = session.run([train, loss], feed_dict={x: x_value, y: y_value})
        errors.append(error_value)
        #print(error_value)
    w_value = session.run(w)
    b_value = session.run(b)
    print('w_value rank: ', tf.rank(w_value), tf.rank(w_value).eval())
    print(type(w_value), type(b_value))
    print(w, b)
    print(w_value, b_value)
    print('%.3f * x0 + %.3f * x1 + %.3f * x2 + %.3f' % (w_value[0], w_value[1], w_value[2], b.eval()))

import matplotlib.pyplot as plt
#print(errors)
plt.plot([np.mean(errors[i: i+20]) for i in range(len(errors) - 50)])
plt.show()
plt.savefig("errors.png")