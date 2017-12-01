import argparse
from constants import *
from os import listdir
import cv2

def main(voc_test_path, results_path):
	all_images_file = '{0}/{1}'.format(voc_test_path, ALL_TEST_IMAGES_FILE_REL)
	test_images_dir = '{0}/{1}'.format(voc_test_path, INPUT_IMAGES_DIR_REL)
	results_images_dir = '{0}/{1}'.format(results_path, RESULTS_IMAGES_DIR_REL)
	with open(all_images_file, 'rb') as f:
		for line in f:
			test_img_path = '{0}/{1}.jpg'.format(test_images_dir, line.strip())
			test_img = cv2.imread(test_img_path)
			result_img_path = '{0}/{1}.png'.format(results_images_dir, line.strip())
			result_img = cv2.imread(result_img_path)
			# print test_img_path, result_img_path
			print test_img.shape, result_img.shape
			if not test_img.shape == result_img.shape:
				print 'Found inconsistency in image', line.strip()
				break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_test_path', help='Path to VOCtest')
	parser.add_argument('results_path')
	args = parser.parse_args()
	main(args.voc_test_path, args.results_path)
