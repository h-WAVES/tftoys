import tensorflow as tf

a = tf.placeholder("float", None)
b = tf.placeholder("float", None)

y = tf.mul(a, b)
ym = tf.matmul(a, b)
diag = tf.diag(a)
mat_inverse = tf.matrix_inverse(tf.matmul(a, b))

model = tf.initialize_all_variables()

sess = tf.Session()
sess.run(model)

print(sess.run(y, feed_dict={a: 3, b: 3}))
print(sess.run(y, feed_dict={a: [3], b: [3]}))
print(sess.run(y, feed_dict={a: [3, 1], b: [3, 1]}))
print(sess.run(y, feed_dict={a: [3, 1, 2], b: [3, 1, 4]}))
print(sess.run(y, feed_dict={a: [[2,3], [1,4]], b: [[1,2], [2,3]]}))
print(sess.run(y, feed_dict={a: [[2,3], [1,4]], b: [[1,2], [2,3]]})) #not a matric operation
#do matrix multipy operation , ym11 = 2*1 + 3*2 = 8, ym22 = 1*2 + 4*3 = 14
m = sess.run(ym, feed_dict={a: [[2,3], [1,4]], b: [[1,2], [2,3]]})
print(m, type(m))
#print(sess.run(diag, feed_dict={a:[[2, 3], [4,5]]}))
minverse = tf.matrix_inverse(m)
print(minverse, type(minverse))
print(sess.run(mat_inverse, feed_dict={a:[[1,2], [2,3]], b:[[2,1], [1,1]]}))
print(sess.run(ym, feed_dict={a: [[2,3], [1,4]], b: [[1,2], [2,3]]}))

sess.close()