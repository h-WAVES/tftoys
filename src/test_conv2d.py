import  tensorflow as tf

def test1():
    input = tf.Variable(tf.random_normal([1,2,2,1]))
    filter = tf.Variable(tf.random_normal([1,1,1,1]))

    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        print("input")
        print(input.eval())
        print("filter")
        print(filter.eval())
        print("result")
        result = sess.run(op)
        print(result)

#test1()

def test2():
    input = tf.Variable(tf.random_normal([1,8,8,1]))
    filter = tf.Variable(tf.random_normal([5,5,1,1]))

    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        print("input")
        print(input.eval())
        print("filter")
        print(filter.eval())
        print("result")
        result = sess.run(op)
        print(result)
test2()