import os
import tensorflow as tf
import argparse
from PIL import Image


ALL_IMAGES_FILE_REL = 'VOC2012/ImageSets/Segmentation/trainval.txt'
IMAGES_DIR_REL = 'VOC2012/JPEGImages'
BATCH_SIZE = 20


def main(voc_devkit_path):
	all_images_file = '{0}/{1}'.format(voc_devkit_path, ALL_IMAGES_FILE_REL)
	images_dir = '{0}/{1}'.format(voc_devkit_path, IMAGES_DIR_REL)

	sorted_images = get_sorted_images(images_dir)
	for i in range(10):
		sorted_images[i].show()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path')
	args = parser.parse_args()
	main(args.voc_devkit_path)
