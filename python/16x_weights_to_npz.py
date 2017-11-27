import argparse
import numpy as np
import tensorflow as tf
from constants import *

def main(meta_file, saved_32x_weights):
	# Restore parameters from 32x saved weights file
	sess = tf.Session()
	saver = tf.train.import_meta_graph(meta_file)
	saver.restore(sess, saved_32x_weights)

	# Pull tensors from saved file
	graph = tf.get_default_graph()

	variable_names = ['conv1_1_W',
					  'conv1_1_b',
					  'conv1_2_W',
					  'conv1_2_b',
					  'conv2_1_W',
					  'conv2_1_b',
					  'conv2_2_W',
					  'conv2_2_b',
					  'conv3_1_W',
					  'conv3_1_b',
					  'conv3_2_W',
					  'conv3_2_b',
					  'conv3_3_W',
					  'conv3_3_b',
					  'conv4_1_W',
					  'conv4_1_b',
					  'conv4_2_W',
					  'conv4_2_b',
					  'conv4_3_W',
					  'conv4_3_b',
					  'conv5_1_W',
					  'conv5_1_b',
					  'conv5_2_W',
					  'conv5_2_b',
					  'conv5_3_W',
					  'conv5_3_b',
					  'conv6_W',
					  'conv6_b',
					  'conv7_W',
					  'conv7_b',
					  'downsampled_32x_W',
					  'downsampled_32x_b',
					  'upsampling_32_to_16_W',
					  'upsampling_32_to_16_b',
					  'pool4_predictions_W',
					  'pool4_predictions_b']

	variable_tensors = [graph.get_tensor_by_name(name + ':0') for name in variable_names]
	variable_weights = sess.run(variable_tensors)

	variable_dict = {}
	for i in range(len(variable_names)):
		variable_dict[variable_names[i]] = variable_weights[i]

	np.savez('saved_16x_weights', **variable_dict)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('meta_file', help='Path to the meta file (likely initial_model/initial_16x_model.meta')
	parser.add_argument('saved_32x_weights', help='Path to saved 32x weights')
	args = parser.parse_args()
	main(args.meta_file, args.saved_32x_weights)