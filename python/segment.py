import os
import argparse
import bilinear_upsampling_weights as bilinear
import color_label_map as clm
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import random
import cv2
from collections import deque
from functools import partial
from PIL import Image

BATCH_SIZE = 10
NUM_CHANNELS = 3
FULLY_CONV_DEPTH = 1024 # Or is it???

MANUAL_VOID_RGB = [1, 1, 1]

ALL_IMAGES_FILE_REL = 'VOC2012/ImageSets/Segmentation/trainval.txt'
INPUT_IMAGES_DIR_REL = 'VOC2012/JPEGImages'
LABEL_IMAGES_DIR_REL = 'VOC2012/SegmentationClass'


def main(voc_devkit_path, index_file):
	image_width = tf.placeholder(tf.int32)
	image_height = tf.placeholder(tf.int32)
	images = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, NUM_CHANNELS])
	ground_truth = tf.placeholder(tf.int32, [BATCH_SIZE, None, None])
	keep_prob = tf.placeholder(tf.float32)

	# Layer 1
	conv1_1_W = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
	conv1_1_b = tf.Variable(tf.truncated_normal([64], stddev=0.1))
	conv1_1 = tf.nn.conv2d(images, conv1_1_W, [1, 1, 1, 1], "SAME") + conv1_1_b

	relu1_1 = tf.nn.relu(conv1_1)

	conv1_2_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
	conv1_2_b = tf.Variable(tf.truncated_normal([64], stddev=0.1))
	conv1_2 = tf.nn.conv2d(relu1_1, conv1_2_W, [1, 1, 1, 1], "SAME") + conv1_2_b

	relu1_2 = tf.nn.relu(conv1_2)

	# Pool 1
	pool1 = tf.nn.max_pool(relu1_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

	# Layer 2
	conv2_1_W = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
	conv2_1_b = tf.Variable(tf.truncated_normal([128], stddev=0.1))
	conv2_1 = tf.nn.conv2d(pool1, conv2_1_W, [1, 1, 1, 1], "SAME") + conv2_1_b

	relu2_1 = tf.nn.relu(conv2_1)

	conv2_2_W = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
	conv2_2_b = tf.Variable(tf.truncated_normal([128], stddev=0.1))
	conv2_2 = tf.nn.conv2d(relu2_1, conv2_2_W, [1, 1, 1, 1], "SAME") + conv2_2_b

	relu2_2 = tf.nn.relu(conv2_2)

	# Pool 2
	pool2 = tf.nn.max_pool(relu2_2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

	# Layer 3
	conv3_1_W = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
	conv3_1_b = tf.Variable(tf.truncated_normal([256], stddev=0.1))
	conv3_1 = tf.nn.conv2d(pool2, conv3_1_W, [1, 1, 1, 1], "SAME") + conv3_1_b

	relu3_1 = tf.nn.relu(conv3_1)

	conv3_2_W = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
	conv3_2_b = tf.Variable(tf.truncated_normal([256], stddev=0.1))
	conv3_2 = tf.nn.conv2d(relu3_1, conv3_2_W, [1, 1, 1, 1], "SAME") + conv3_2_b

	relu3_2 = tf.nn.relu(conv3_2)

	conv3_3_W = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
	conv3_3_b = tf.Variable(tf.truncated_normal([256], stddev=0.1))
	conv3_3 = tf.nn.conv2d(relu3_2, conv3_3_W, [1, 1, 1, 1], "SAME") + conv3_3_b

	relu3_3 = tf.nn.relu(conv3_3)

	# Pool 3
	pool3 = tf.nn.max_pool(relu3_3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

	# Layer 4
	conv4_1_W = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
	conv4_1_b = tf.Variable(tf.truncated_normal([512], stddev=0.1))
	conv4_1 = tf.nn.conv2d(pool3, conv4_1_W, [1, 1, 1, 1], "SAME") + conv4_1_b

	relu4_1 = tf.nn.relu(conv4_1)

	conv4_2_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
	conv4_2_b = tf.Variable(tf.truncated_normal([512], stddev=0.1))
	conv4_2 = tf.nn.conv2d(relu4_1, conv4_2_W, [1, 1, 1, 1], "SAME") + conv4_2_b

	relu4_2 = tf.nn.relu(conv4_2)

	conv4_3_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
	conv4_3_b = tf.Variable(tf.truncated_normal([512], stddev=0.1))
	conv4_3 = tf.nn.conv2d(relu4_2, conv4_3_W, [1, 1, 1, 1], "SAME") + conv4_3_b

	relu4_3 = tf.nn.relu(conv4_3)

	# Pool 4
	pool4 = tf.nn.max_pool(relu4_3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

	# Layer 5
	conv5_1_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
	conv5_1_b = tf.Variable(tf.truncated_normal([512], stddev=0.1))
	conv5_1 = tf.nn.conv2d(pool4, conv5_1_W, [1, 1, 1, 1], "SAME") + conv5_1_b

	relu5_1 = tf.nn.relu(conv5_1)

	conv5_2_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
	conv5_2_b = tf.Variable(tf.truncated_normal([512], stddev=0.1))
	conv5_2 = tf.nn.conv2d(relu5_1, conv5_2_W, [1, 1, 1, 1], "SAME") + conv5_2_b

	relu5_2 = tf.nn.relu(conv5_2)

	conv5_3_W = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
	conv5_3_b = tf.Variable(tf.truncated_normal([512], stddev=0.1))
	conv5_3 = tf.nn.conv2d(relu5_2, conv5_3_W, [1, 1, 1, 1], "SAME") + conv5_3_b

	relu5_3 = tf.nn.relu(conv5_3)

	# Pool 5
	pool5 = tf.nn.max_pool(relu5_3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

	# Layer 6
	conv6_W = tf.Variable(tf.truncated_normal([1, 1, 512, FULLY_CONV_DEPTH], stddev=0.1))
	conv6_b = tf.Variable(tf.truncated_normal([FULLY_CONV_DEPTH], stddev=0.1))

	conv6 = tf.nn.conv2d(pool5, conv6_W, [1, 1, 1, 1], "SAME") + conv6_b

	relu6 = tf.nn.relu(conv6)
	drop6 = tf.nn.dropout(relu6, keep_prob)

	# Layer 7
	conv7_W = tf.Variable(tf.truncated_normal([1, 1, FULLY_CONV_DEPTH, FULLY_CONV_DEPTH], stddev=0.1))
	conv7_b = tf.Variable(tf.truncated_normal([FULLY_CONV_DEPTH], stddev=0.1))

	conv7 = tf.nn.conv2d(drop6, conv7_W, [1, 1, 1, 1], "SAME") + conv7_b

	relu7 = tf.nn.relu(conv7)
	drop7 = tf.nn.dropout(relu7, keep_prob)

	# Calculate 32x downsampled predictions
	downsampled_32x_W = tf.Variable(tf.truncated_normal([1, 1, FULLY_CONV_DEPTH, 21], stddev=0.1))
	downsampled_32x_b = tf.Variable(tf.truncated_normal([21], stddev=0.1))

	downsampled_32x = tf.nn.conv2d(drop7, downsampled_32x_W, [1, 1, 1, 1], "SAME") + downsampled_32x_b

	# Upsample 32x
	upsampling_W = tf.Variable(bilinear.create_initial_bilinear_weights(64))
	upsampling_b = tf.Variable(tf.truncated_normal([21], stddev=0.1))

	upsampled_output = tf.nn.conv2d_transpose(downsampled_32x, upsampling_W, [BATCH_SIZE, image_height, image_width, 21], [1, 32, 32, 1]) + upsampling_b

	# Loss and mean_iou calculation
	void_pixel_mask = tf.cast(tf.not_equal(ground_truth, 255), tf.int32)
	ground_truth_without_void = tf.multiply(ground_truth, void_pixel_mask)

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=upsampled_output, labels=ground_truth_without_void))
	train = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)

	predictions = tf.argmax(upsampled_output, axis=3)
	mean_iou, _ = tf.metrics.mean_iou(ground_truth, predictions, 21, weights=void_pixel_mask)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	label_to_color, color_to_label = clm.create_color_label_map()

	# Training training	
	all_images_file = '{0}/{1}'.format(voc_devkit_path, ALL_IMAGES_FILE_REL)
	input_images_dir = '{0}/{1}'.format(voc_devkit_path, INPUT_IMAGES_DIR_REL)
	label_images_dir = '{0}/{1}'.format(voc_devkit_path, LABEL_IMAGES_DIR_REL)

	sorted_img_names = []
	with open(index_file, 'rb') as f:
		sorted_img_names = [line[:-1] for line in f]
	num_imgs = len(sorted_img_names)

	for i in range(20):
 		print i
 		r = random.randint(0, num_imgs - BATCH_SIZE)
 		name_batch = sorted_img_names[r : r + BATCH_SIZE]
		input_batch, label_batch = pad_batch(name_batch, input_images_dir, label_images_dir)

		label_batch = map(lambda label_img: clm.rgb_image_to_label(np.array(label_img, dtype='uint8'), color_to_label), label_batch)
		width, height, channels = input_batch[0].shape

 		_, l = sess.run([train, loss], feed_dict={images: input_batch, ground_truth: label_batch, image_width: width, image_height: height, keep_prob: 0.5})
 		print l

 	# Testing testing


def pad_img(H, W, border_type, img):
	h = img.shape[0]
	w = img.shape[1]

	top = (H - h) / 2
	bottom = H - h - top
	left = (W - w) / 2
	right = W - w - left

	return cv2.copyMakeBorder(img, top, bottom, left, right, border_type, value=MANUAL_VOID_RGB)


def read_img_rgb(img_path):
	return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def pad_batch(name_batch, input_images_dir, label_images_dir):
	input_batch_raw = map(lambda img_name: read_img_rgb('{0}/{1}.jpg'.format(input_images_dir, img_name)), name_batch)
	label_batch_raw = map(lambda img_name: read_img_rgb('{0}/{1}.png'.format(label_images_dir, img_name)), name_batch)
	
	H = max([img.shape[0] for img in input_batch_raw])
	W = max([img.shape[1] for img in input_batch_raw])

	padded_inputs = map(partial(pad_img, H, W, cv2.BORDER_REPLICATE), input_batch_raw)
	padded_labels = map(partial(pad_img, H, W, cv2.BORDER_CONSTANT), label_batch_raw)

	return padded_inputs, padded_labels


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path', help='Path to VOCdevkit directory')
	parser.add_argument('index_file', help='Path to images index file (likely data/images_index.txt)')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.index_file)
