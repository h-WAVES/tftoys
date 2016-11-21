import tensorflow as tf

w = tf.Variable(tf.random_uniform([3], -1.0, 1.0))
cc = tf.constant(5)
c = tf.constant([2,3])
cm = tf.constant([[1,2],[2,3]])

c3m = tf.constant([[[1,2], [2,3]], [[3,4], [5,6]]])
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(w)
print(tf.rank(sess.run(w)).eval(session=sess))

print(cc)
ccr = sess.run(cc)
print(ccr, tf.rank(ccr))
print(tf.rank(ccr).eval(session=sess))

print(c)
cr = sess.run(c)
print(cr, tf.rank(cr))
print(tf.rank(cr).eval(session=sess))

cmr = sess.run(cm)
print(cmr, tf.rank(cmr))
print(tf.rank(cmr).eval(session=sess))

c3mr = sess.run(c3m)
print(c3mr, tf.rank(c3mr))
print(tf.rank(c3mr).eval(session=sess))
sess.close()