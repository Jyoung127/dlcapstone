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
from train_segmentation_net import pad_img, read_img_rgb, pad_batch

def main(voc_devkit_path, index_file, meta_file, saved_weights):
	# Restore old session from saved file
	sess = tf.Session()
	saver = tf.train.import_meta_graph(meta_file)
	saver.restore(sess, saved_weights)

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

	for i in range(1):
 		r = random.randint(0, num_imgs - BATCH_SIZE)
 		name_batch = sorted_img_names[r : r + BATCH_SIZE]
		input_batch, label_batch = pad_batch(name_batch, input_images_dir, label_images_dir)

		label_batch = map(lambda label_img: clm.rgb_image_to_label(np.array(label_img, dtype='uint8'), color_to_label), label_batch)
		height, width, channels = input_batch[0].shape

		feed_dict = {images: input_batch, ground_truth: label_batch, image_width: width, image_height: height, keep_prob: 0.5}

 		# preds, miou = sess.run([predictions, mean_iou], feed_dict)
 		preds = sess.run(predictions, feed_dict)

 		prediction_images = map(lambda pred_img: clm.label_image_to_rgb(pred_img, label_to_color), preds)

 		for j in range(BATCH_SIZE):
 			predicted_image = prediction_images[j]
 			image_filename = '{0}.png'.format(name_batch[j])
 			cv2.imwrite(image_filename, cv2.cvtColor(np.uint8(predicted_image), cv2.COLOR_RGB2BGR))

 		# print 'mean iou is', miou


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path', help='Path to VOCdevkit directory')
	parser.add_argument('index_file', help='Path to images index file (likely data/images_index.txt)')
	parser.add_argument('meta_file', help='Path to the meta file (likely initial_model/initial_32x_model.meta')
	parser.add_argument('saved_weights', help='Path to saved weights directory (e.g. saved_weights/saved_32x_weights-9)')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.index_file, args.meta_file, args.saved_weights)
