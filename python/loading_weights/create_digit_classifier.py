import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz = 10
img = tf.placeholder(tf.float32, [None, 784], name="img")
ans = tf.placeholder(tf.float32, [None, 10], name="ans")

U = tf.Variable(tf.random_normal([784, 10], stddev=.1), name="U")
bU = tf.Variable(tf.random_normal([10], stddev=.1), name="bU")

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

sess.run(tf.global_variables_initializer())

logits = tf.matmul(img, U) + bU
l1_output = tf.nn.relu(logits)
prbs = tf.nn.softmax(l1_output)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, name="train_step")

correct_prediction = tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

saver.save(sess, 'test_model')
