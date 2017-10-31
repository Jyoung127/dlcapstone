import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz=10
img = tf.placeholder(tf.float32, [None, 784])
ans = tf.placeholder(tf.float32, [None, 10])

U = tf.Variable(tf.random_normal([784, 10], stddev=.1))
bU = tf.Variable(tf.random_normal([10], stddev=.1))

sess = tf.Session()
#saver=tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
saver = tf.train.import_meta_graph('test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
sess.run(tf.global_variables_initializer())

logits = tf.matmul(img, U) + bU
l1_output = tf.nn.relu(logits)
prbs = tf.nn.softmax(l1_output)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for _ in range(10):
	img_batch, ans_batch = mnist.train.next_batch(batchSz)
	sess.run(train_step, feed_dict={img: img_batch, ans: ans_batch})

correct_prediction = tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={img: mnist.test.images, ans: mnist.test.labels}))
saver.save(sess, 'test_model')
