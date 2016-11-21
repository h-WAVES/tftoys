import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np

num_puntos = 2000
conjunto_puntos = []
for i in xrange(num_puntos):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

df = pd.DataFrame({'x': [v[0] for v in conjunto_puntos],
                   "y": [v[1] for v in conjunto_puntos]})
#sb.lmplot('x', 'y', data=df, fit_reg=False, size=6)
#plt.show()
vectors = tf.constant(conjunto_puntos)
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k-1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2), 0)
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])),
                                     reduction_indices=[1]) for c in xrange(k)])
update_centroides = tf.assign(centroids, means)
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

