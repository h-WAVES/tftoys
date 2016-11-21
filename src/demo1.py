import tensorflow as tf

a = tf.placeholder("float")
# a = tf.placeholder("float", None) # alternative
b = tf.placeholder("float")

y = tf.mul(a, b)

#model = tf.initialize_all_variables()

sess = tf.Session()

print(sess.run(y, feed_dict={a: 3, b: 3}))
print(sess.run(y, feed_dict={a: [3], b: [3]}))
print(sess.run(y, feed_dict={a: [3, 1], b: [3, 1]}))
print(sess.run(y, feed_dict={a: [3, 1, 2], b: [3, 1, 4]}))
print(sess.run(y, feed_dict={a: [[2,3], [1,4]], b: [[1,2], [2,3]]}))
print(sess.run(y, feed_dict={a: [[2,3], [1,4]], b: [[1,2], [2,3]]})) #not a matric operation

sess.close()
