import os
from sets import Set

ALL_IMAGES_FILE = 'VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'
IMAGES_DIR = 'VOCdevkit/VOC2012/JPEGImages'

def main():
	with open(ALL_IMAGES_FILE) as images_file:
		images = Set([line[:-1] for line in images_file.readlines()])

	files = os.listdir(IMAGES_DIR)

	for image_file in files:
		if image_file[:-4] not in images:
			os.remove(IMAGES_DIR + "/" + image_file)

if __name__ == '__main__':
	main()