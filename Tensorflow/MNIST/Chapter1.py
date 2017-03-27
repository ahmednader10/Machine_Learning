import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

X = tf.reshape(X, [-1, 784])
#model
Y = tf.nn.softmax(tf.matmul(X, W) + b)

#placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

#loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    #load batch images and correct images
    batch_X, batch_Y = mnist.train.next_batch(100)

    train_data = {X: batch_X, Y_: batch_Y}
    #train
    sess.run(train_step, feed_dict = {X: batch_X, Y_: batch_Y})

    #print in case of success
    a,c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})

    #success on test data?
    test_data = {X:mnist.test.images, Y_:mnist.test.labels}
    a,c = sess.run([accuracy, cross_entropy], feed_dict = {X:mnist.test.images, Y_:mnist.test.labels})

print("accuracy:" + str(a) + " loss: " + str(c))
