import argparse
import bilinear_upsampling_weights as bilinear
import color_label_map as clm
import numpy as np
import tensorflow as tf
from constants import *

def main(saved_32x_weights):
	npz = np.load(saved_32x_weights)

	image_width = tf.placeholder(tf.int32, name='image_width')
	image_height = tf.placeholder(tf.int32, name='image_height')
	images = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, NUM_CHANNELS], name='images')
	ground_truth = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], name='ground_truth')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	# Layer 1
	conv1_1_W = tf.Variable(npz['conv1_1_W'], name='conv1_1_W')
	conv1_1_b = tf.Variable(npz['conv1_1_b'], name='conv1_1_b')
	conv1_1 = tf.add(tf.nn.conv2d(images, conv1_1_W, [1, 1, 1, 1], "SAME"), conv1_1_b, name='conv1_1')

	relu1_1 = tf.nn.relu(conv1_1, name='relu1_1')

	conv1_2_W = tf.Variable(npz['conv1_2_W'], name='conv1_2_W')
	conv1_2_b = tf.Variable(npz['conv1_2_b'], name='conv1_2_b')
	conv1_2 = tf.add(tf.nn.conv2d(relu1_1, conv1_2_W, [1, 1, 1, 1], "SAME"), conv1_2_b, name='conv1_2')

	relu1_2 = tf.nn.relu(conv1_2, name='relu1_2')

	# Pool 1
	pool1 = tf.nn.max_pool(relu1_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name='pool1')

	# Layer 2
	conv2_1_W = tf.Variable(npz['conv2_1_W'], name='conv2_1_W')
	conv2_1_b = tf.Variable(npz['conv2_1_b'], name='conv2_1_b')
	conv2_1 = tf.add(tf.nn.conv2d(pool1, conv2_1_W, [1, 1, 1, 1], "SAME"), conv2_1_b, name='conv2_1')

	relu2_1 = tf.nn.relu(conv2_1, name='relu2_1')

	conv2_2_W = tf.Variable(npz['conv2_2_W'], name='conv2_2_W')
	conv2_2_b = tf.Variable(npz['conv2_2_b'], name='conv2_2_b')
	conv2_2 = tf.add(tf.nn.conv2d(relu2_1, conv2_2_W, [1, 1, 1, 1], "SAME"), conv2_2_b, name='conv2_2')

	relu2_2 = tf.nn.relu(conv2_2, name='relu2_2')

	# Pool 2
	pool2 = tf.nn.max_pool(relu2_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name='pool2')

	# Layer 3
	conv3_1_W = tf.Variable(npz['conv3_1_W'], name='conv3_1_W')
	conv3_1_b = tf.Variable(npz['conv3_1_b'], name='conv3_1_b')
	conv3_1 = tf.add(tf.nn.conv2d(pool2, conv3_1_W, [1, 1, 1, 1], "SAME"), conv3_1_b, name='conv3_1')

	relu3_1 = tf.nn.relu(conv3_1, name='relu3_1')

	conv3_2_W = tf.Variable(npz['conv3_2_W'], name='conv3_2_W')
	conv3_2_b = tf.Variable(npz['conv3_2_b'], name='conv3_2_b')
	conv3_2 = tf.add(tf.nn.conv2d(relu3_1, conv3_2_W, [1, 1, 1, 1], "SAME"), conv3_2_b, name='conv3_2')

	relu3_2 = tf.nn.relu(conv3_2, name='relu3_2')

	conv3_3_W = tf.Variable(npz['conv3_3_W'], name='conv3_3_W')
	conv3_3_b = tf.Variable(npz['conv3_3_b'], name='conv3_3_b')
	conv3_3 = tf.add(tf.nn.conv2d(relu3_2, conv3_3_W, [1, 1, 1, 1], "SAME"), conv3_3_b, name='conv3_3')

	relu3_3 = tf.nn.relu(conv3_3, name='relu3_3')

	# Pool 3
	pool3 = tf.nn.max_pool(relu3_3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name='pool3')

	# Layer 4
	conv4_1_W = tf.Variable(npz['conv4_1_W'], name='conv4_1_W')
	conv4_1_b = tf.Variable(npz['conv4_1_b'], name='conv4_1_b')
	conv4_1 = tf.add(tf.nn.conv2d(pool3, conv4_1_W, [1, 1, 1, 1], "SAME"), conv4_1_b, name='conv4_1')

	relu4_1 = tf.nn.relu(conv4_1, name='relu4_1')

	conv4_2_W = tf.Variable(npz['conv4_2_W'], name='conv4_2_W')
	conv4_2_b = tf.Variable(npz['conv4_2_b'], name='conv4_2_b')
	conv4_2 = tf.add(tf.nn.conv2d(relu4_1, conv4_2_W, [1, 1, 1, 1], "SAME"), conv4_2_b, name='conv4_2')

	relu4_2 = tf.nn.relu(conv4_2, name='relu4_2')

	conv4_3_W = tf.Variable(npz['conv4_3_W'], name='conv4_3_W')
	conv4_3_b = tf.Variable(npz['conv4_3_b'], name='conv4_3_b')
	conv4_3 = tf.add(tf.nn.conv2d(relu4_2, conv4_3_W, [1, 1, 1, 1], "SAME"), conv4_3_b, name='conv4_3')

	relu4_3 = tf.nn.relu(conv4_3, name='relu4_3')

	# Pool 4
	pool4 = tf.nn.max_pool(relu4_3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name='pool4')

	# Layer 5
	conv5_1_W = tf.Variable(npz['conv5_1_W'], name='conv5_1_W')
	conv5_1_b = tf.Variable(npz['conv5_1_b'], name='conv5_1_b')
	conv5_1 = tf.add(tf.nn.conv2d(pool4, conv5_1_W, [1, 1, 1, 1], "SAME"), conv5_1_b, name='conv5_1')

	relu5_1 = tf.nn.relu(conv5_1, name='relu5_1')

	conv5_2_W = tf.Variable(npz['conv5_2_W'], name='conv5_2_W')
	conv5_2_b = tf.Variable(npz['conv5_2_b'], name='conv5_2_b')
	conv5_2 = tf.add(tf.nn.conv2d(relu5_1, conv5_2_W, [1, 1, 1, 1], "SAME"), conv5_2_b, name='conv5_2')

	relu5_2 = tf.nn.relu(conv5_2, name='relu5_2')

	conv5_3_W = tf.Variable(npz['conv5_3_W'], name='conv5_3_W')
	conv5_3_b = tf.Variable(npz['conv5_3_b'], name='conv5_3_b')
	conv5_3 = tf.add(tf.nn.conv2d(relu5_2, conv5_3_W, [1, 1, 1, 1], "SAME"), conv5_3_b, name='conv5_3')

	relu5_3 = tf.nn.relu(conv5_3, name='relu5_3')

	# Pool 5
	pool5 = tf.nn.max_pool(relu5_3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name='pool5')

	# Layer 6
	conv6_W = tf.Variable(npz['conv6_W'], name='conv6_W')
	conv6_b = tf.Variable(npz['conv6_b'], name='conv6_b')

	conv6 = tf.add(tf.nn.conv2d(pool5, conv6_W, [1, 1, 1, 1], "SAME"), conv6_b, name='conv6')

	relu6 = tf.nn.relu(conv6, name='relu6')
	drop6 = tf.nn.dropout(relu6, keep_prob, name='drop6')

	# Layer 7
	conv7_W = tf.Variable(npz['conv7_W'], name='conv7_W')
	conv7_b = tf.Variable(npz['conv7_b'], name='conv7_b')

	conv7 = tf.add(tf.nn.conv2d(drop6, conv7_W, [1, 1, 1, 1], "SAME"), conv7_b, name='conv7')

	relu7 = tf.nn.relu(conv7, name='relu7')
	drop7 = tf.nn.dropout(relu7, keep_prob, name='drop7')

	# Calculate 32x downsampled predictions
	downsampled_32x_W = tf.Variable(npz['downsampled_32x_W'], name='downsampled_32x_W')
	downsampled_32x_b = tf.Variable(npz['downsampled_32x_b'], name='downsampled_32x_b')

	downsampled_32x = tf.add(tf.nn.conv2d(drop7, downsampled_32x_W, [1, 1, 1, 1], "SAME"), downsampled_32x_b, name='downsampled_32x')

	# Upsample 32x prediction layer to 16x layer
	upsampling_32_to_16_W = tf.Variable(bilinear.create_initial_bilinear_weights(4), name='upsampling_32_to_16_W')
	upsampling_32_to_16_b = tf.Variable(tf.truncated_normal([21], stddev=0.1), name='upsampling_32_to_16_b')

	upsampled_from_32x = tf.add(tf.nn.conv2d_transpose(downsampled_32x, upsampling_32_to_16_W, [BATCH_SIZE, stride_16(image_height), stride_16(image_width), 21], [1, 2, 2, 1]), upsampling_32_to_16_b, name='upsampled_from_32x')

	# Create coarse predictions from pool4 layer
	pool4_predictions_W = tf.Variable(tf.zeros([1, 1, 512, 21]), name='pool4_predictions_W')
	pool4_predictions_b = tf.Variable(tf.zeros([21]), name='pool4_predictions_b')

	pool4_predictions = tf.add(tf.nn.conv2d(pool4, pool4_predictions_W, [1, 1, 1, 1], "SAME"), pool4_predictions_b, name='pool4_predictions')

	# Combine both 16x predictions
	downsampled_16x = tf.add(upsampled_from_32x, pool4_predictions)

	# Upsample 16x
	upsampling_16x_W = tf.Variable(bilinear.create_initial_bilinear_weights(32), name='upsampling_16x_W')
	upsampling_16x_b = tf.Variable(tf.truncated_normal([21], stddev=0.1), name='upsampling_16x_b')

	upsampled_16x_output = tf.add(tf.nn.conv2d_transpose(downsampled_16x, upsampling_16x_W, [BATCH_SIZE, image_height, image_width, 21], [1, 16, 16, 1]), upsampling_16x_b, name='upsampled_16x_output')

	# Loss and mean_iou calculation
	void_pixel_mask = tf.cast(tf.not_equal(ground_truth, 255), tf.int32, name='void_pixel_mask')
	border_pixel_mask = tf.cast(tf.not_equal(ground_truth, clm.MANUAL_VOID_LABEL), tf.int32, name='border_pixel_mask')
	full_mask = tf.multiply(void_pixel_mask, border_pixel_mask, name='full_mask')
	ground_truth_without_void = tf.multiply(ground_truth, full_mask, name='ground_truth_without_void')

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=upsampled_16x_output, labels=ground_truth_without_void), name='loss')
	train = tf.train.AdamOptimizer(LEARNING_RATE * 0.01).minimize(loss, name='train')

	predictions = tf.argmax(upsampled_16x_output, axis=3, name='predictions')
	masked_ground_truth = tf.multiply(ground_truth, full_mask)
	mean_iou, _ = tf.metrics.mean_iou(masked_ground_truth, predictions, 21, weights=full_mask)

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
	saver.save(sess, 'initial_model/initial_16x_model')

def stride_16(n):
	stride_2 = (n + 1) / 2
	stride_4 = (stride_2 + 1) / 2
	stride_8 = (stride_4 + 1) / 2
	stride_16 = (stride_8 + 1) / 2
	return stride_16

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('saved_32x_weights', help='Path to saved 32x weights')
	args = parser.parse_args()
	main(args.saved_32x_weights)