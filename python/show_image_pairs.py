import argparse
import cv2
import numpy as np
import os
import sys
from constants import *

def main(voc_devkit_path, image_paths):
	print 'Commands (when image window is selected):'
	print '    n: next image'
	print '    b: previous image'
	print '    q: quit'

	original_images_dir = '{0}/{1}'.format(voc_devkit_path, INPUT_IMAGES_DIR_REL)
	ground_truth_dir = '{0}/{1}'.format(voc_devkit_path, LABEL_IMAGES_DIR_REL)

	cv2.namedWindow('original_image_window')
	cv2.namedWindow('ground_truth_window')
	cv2.namedWindow('image_window')

	current_idx = 0

	while True:
		image_path = image_paths[current_idx]
		image_file = os.path.basename(image_path)

		original_image_path = '{0}/{1}.jpg'.format(original_images_dir, image_file[:-4])
		ground_truth_path = '{0}/{1}.png'.format(ground_truth_dir, image_file[:-4])

		original_image = cv2.imread(original_image_path)
		ground_truth = cv2.imread(ground_truth_path)
		image = cv2.imread(image_path)

		cv2.imshow('original_image_window', np.uint8(original_image))
		cv2.imshow('ground_truth_window', np.uint8(ground_truth))
		cv2.imshow('image_window', np.uint8(image))

		key = cv2.waitKey()
		if key == ord('n'):
			current_idx = min(current_idx + 1, len(image_paths) - 1)
		elif key == ord('b'):
			current_idx = max(current_idx - 1 , 0)
		elif key == ord('q'):
			break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path', help='Path to VOCdevkit directory')
	parser.add_argument('image_paths', help='Images to compare to ground truth', nargs='+')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.image_paths)