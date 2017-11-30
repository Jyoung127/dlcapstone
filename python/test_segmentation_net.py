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
from train_32x_segmentation_net import pad_img, read_img_rgb

def main(voc_devkit_path, index_file, meta_file, saved_weights, results_path):
	# Restore old session from saved file
	sess = tf.Session()
	saver = tf.train.import_meta_graph(meta_file)
	saver.restore(sess, saved_weights)

	sess.run(tf.local_variables_initializer())

	# Pull tensors from saved file
	graph = tf.get_default_graph()

	images = graph.get_tensor_by_name("images:0")
	image_width = graph.get_tensor_by_name("image_width:0")
	image_height = graph.get_tensor_by_name("image_height:0")
	ground_truth = graph.get_tensor_by_name("ground_truth:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")
	predictions = graph.get_tensor_by_name("predictions:0")
	full_mask = graph.get_tensor_by_name("full_mask:0")
	mean_iou = graph.get_tensor_by_name("mean_iou/mean_iou:0")
	loss = graph.get_tensor_by_name("loss:0")
	update_op = graph.get_operation_by_name("mean_iou/AssignAdd")
	train = graph.get_operation_by_name("train")

	label_to_color, color_to_label = clm.create_color_label_map()

	# Training training	
	all_images_file = '{0}/{1}'.format(voc_devkit_path, ALL_IMAGES_FILE_REL)
	input_images_dir = '{0}/{1}'.format(voc_devkit_path, INPUT_IMAGES_DIR_REL)
	label_images_dir = '{0}/{1}'.format(voc_devkit_path, LABEL_IMAGES_DIR_REL)
	results_dir = '{0}/{1}'.format(results_path, VOC_COMP_DIR_REL)

	sorted_img_names = []
	with open(index_file, 'rb') as f:
		sorted_img_names = [line[:-1] for line in f]
	num_imgs = len(sorted_img_names)

	def run_batch(j, k):
		print 'Processing images from index', j, 'to', k
 		name_batch = sorted_img_names[j : k]
		input_batch = pad_batch(name_batch, input_images_dir)

		height, width, channels = input_batch[0].shape

		feed_dict = {images: input_batch, image_width: width, image_height: height, keep_prob: 1.0}

 		preds = sess.run(predictions, feed_dict)

 		prediction_images = map(lambda pred_img: clm.label_image_to_rgb(pred_img, label_to_color), preds)
 		unpadded_predictions = unpad_batch(name_batch, prediction_images, input_images_dir)

 		for j in range(BATCH_SIZE):
 			predicted_image = unpadded_predictions[j]
 			image_filename = '{0}/{1}.png'.format(results_dir, name_batch[j])
 			cv2.imwrite(image_filename, cv2.cvtColor(np.uint8(predicted_image), cv2.COLOR_RGB2BGR))


	for i in range(num_imgs / BATCH_SIZE):
		run_batch(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)

	run_batch(num_imgs - BATCH_SIZE, num_imgs)



def unpad_img(img_name, prediction_img, input_images_dir):
	original_img = read_img_rgb('{0}/{1}.jpg'.format(input_images_dir, img_name))
	h, w = original_img.shape[0], original_img.shape[1]
	H, W = prediction_img.shape[0], prediction_img.shape[1]

	top = (H - h) / 2
	bottom = H - h - top
	left = (W - w) / 2
	right = W - w - left

	return prediction_img[top : H - bottom, left : W - right]


def unpad_batch(name_batch, prediction_images, input_images_dir):
 	name2pi = {name_batch[i]: prediction_images[i] for i in range(len(name_batch))}
 	return map(lambda img_name: unpad_img(img_name, name2pi[img_name], input_images_dir), name_batch)


def pad_batch(name_batch, input_images_dir):
	input_batch_raw = map(lambda img_name: read_img_rgb('{0}/{1}.jpg'.format(input_images_dir, img_name)), name_batch)
	
	H = max([img.shape[0] for img in input_batch_raw])
	W = max([img.shape[1] for img in input_batch_raw])

	padded_inputs = map(partial(pad_img, H, W, cv2.BORDER_REPLICATE), input_batch_raw)

	return padded_inputs


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path', help='Path to VOCdevkit directory')
	parser.add_argument('index_file', help='Path to images index file (likely data/test_images_index.txt)')
	parser.add_argument('meta_file', help='Path to the meta file (likely initial_model/initial_32x_model.meta')
	parser.add_argument('saved_weights', help='Path to saved weights directory (e.g. saved_weights/saved_32x_weights-9)')
	parser.add_argument('results_path', help='Path to results directory (likely ./results)')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.index_file, args.meta_file, args.saved_weights, args.results_path)
