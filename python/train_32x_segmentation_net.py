import os
import argparse
import bilinear_upsampling_weights as bilinear
import color_label_map as clm
import numpy as np
import tensorflow as tf
import random
import cv2
from collections import deque
from functools import partial
from constants import *

def main(voc_devkit_path, index_file, meta_file, saved_weights_dir):
	# Restore old session from saved file
	sess = tf.Session()
	saver = tf.train.import_meta_graph(meta_file)
	# On first iteration be sure to load from to the initial model weights,
	# but still save to another set of weights to avoid overwriting them!
	saver.restore(sess, tf.train.latest_checkpoint(saved_weights_dir))

	# Pull tensors from saved file
	graph = tf.get_default_graph()

	images = graph.get_tensor_by_name("images:0")
	image_width = graph.get_tensor_by_name("image_width:0")
	image_height = graph.get_tensor_by_name("image_height:0")
	ground_truth = graph.get_tensor_by_name("ground_truth:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")
	predictions = graph.get_tensor_by_name("predictions:0")
	mean_iou = graph.get_tensor_by_name("mean_iou/mean_iou:0")
	loss = graph.get_tensor_by_name("loss:0")
	train = graph.get_operation_by_name("train")

	label_to_color, color_to_label = clm.create_color_label_map()

	# Training training	
	all_images_file = '{0}/{1}'.format(voc_devkit_path, ALL_IMAGES_FILE_REL)
	input_images_dir = '{0}/{1}'.format(voc_devkit_path, INPUT_IMAGES_DIR_REL)
	label_images_dir = '{0}/{1}'.format(voc_devkit_path, LABEL_IMAGES_DIR_REL)

	sorted_img_names = []
	with open(index_file, 'rb') as f:
		sorted_img_names = [line[:-1] for line in f]
	num_imgs = len(sorted_img_names)

	for j in range(48, 101):
		for i in range(NUM_BATCHES):
			r = random.randint(0, num_imgs - BATCH_SIZE)
			name_batch = sorted_img_names[r : r + BATCH_SIZE]
			input_batch, label_batch = pad_batch(name_batch, input_images_dir, label_images_dir)

			label_batch = map(lambda label_img: clm.rgb_image_to_label(np.array(label_img, dtype='uint8'), color_to_label), label_batch)
			height, width, channels = input_batch[0].shape

			feed_dict = {images: input_batch, ground_truth: label_batch, image_width: width, image_height: height, keep_prob: 0.5}

			_, l = sess.run([train, loss], feed_dict)
			print 'loss for batch', i, 'in epoch', j, 'is', l
		saver.save(sess, saved_weights_dir + '/saved_32x_weights_epoch_' + str(j), global_step=j)

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
	parser.add_argument('meta_file', help='Path to the meta file (likely initial_model/initial_32x_model.meta')
	parser.add_argument('saved_weights_dir', help='Path to saved weights directory (likely saved_weights)')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.index_file, args.meta_file, args.saved_weights_dir)
