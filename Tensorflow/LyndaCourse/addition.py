import tensorflow as tf
import os

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="addition")

#create session
with tf.Session() as session:
    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]})

    print(result)
