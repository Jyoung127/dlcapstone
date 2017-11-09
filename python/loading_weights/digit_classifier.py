import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz = 10

sess = tf.Session()
saver = tf.train.import_meta_graph('test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./saved_epochs'))

graph = tf.get_default_graph()

img = graph.get_tensor_by_name("img:0")
ans = graph.get_tensor_by_name("ans:0")
accuracy = graph.get_tensor_by_name("accuracy:0")
train_step = graph.get_operation_by_name("train_step")

for epoch in range(10):
	for batch in range(100):
		img_batch, ans_batch = mnist.train.next_batch(batchSz)
		sess.run(train_step, feed_dict={img: img_batch, ans: ans_batch})

	print sess.run(accuracy, feed_dict={img: mnist.test.images, ans: mnist.test.labels})
	saver.save(sess, 'saved_epochs/epoch_' + str(epoch))
