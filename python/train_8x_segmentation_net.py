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
from train_32x_segmentation_net import pad_img, read_img_rgb, pad_batch

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

	for j in range(1, 101):
		for i in range(NUM_BATCHES):
			r = random.randint(0, num_imgs - BATCH_SIZE)
			name_batch = sorted_img_names[r : r + BATCH_SIZE]
			input_batch, label_batch = pad_batch(name_batch, input_images_dir, label_images_dir)

			label_batch = map(lambda label_img: clm.rgb_image_to_label(np.array(label_img, dtype='uint8'), color_to_label), label_batch)
			height, width, channels = input_batch[0].shape

			feed_dict = {images: input_batch, ground_truth: label_batch, image_width: width, image_height: height, keep_prob: 0.5}

			_, l = sess.run([train, loss], feed_dict)
			print 'loss for batch', i, 'in epoch', j, 'is', l
		saver.save(sess, saved_weights_dir + '/saved_8x_weights_epoch_' + str(j), global_step=j)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path', help='Path to VOCdevkit directory')
	parser.add_argument('index_file', help='Path to images index file (likely data/images_index.txt)')
	parser.add_argument('meta_file', help='Path to the meta file (likely initial_model/initial_8x_model.meta')
	parser.add_argument('saved_weights_dir', help='Path to saved weights directory (likely saved_weights)')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.index_file, args.meta_file, args.saved_weights_dir)
