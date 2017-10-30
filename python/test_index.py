import argparse
import random
from PIL import Image


BATCH_SIZE = 20
NUM_BATCHES = 100000
ALL_IMAGES_FILE_REL = 'VOC2012/ImageSets/Segmentation/trainval.txt'
IMAGES_DIR_REL = 'VOC2012/JPEGImages'


def main(voc_devkit_path, index_file):
	all_images_file = '{0}/{1}'.format(voc_devkit_path, ALL_IMAGES_FILE_REL)
	images_dir = '{0}/{1}'.format(voc_devkit_path, IMAGES_DIR_REL)

	sorted_files = []
	with open(index_file, 'rb') as f:
		sorted_files = [line[:-1] for line in f]

	# f2i = {f: i for i, f in enumerate(sorted_files)}

	# random.shuffle(sorted_files)

	num_files = len(sorted_files)
	pixel_loss = 0
	for i in range(NUM_BATCHES):
		r = random.randint(0, num_files - BATCH_SIZE)
		batch = sorted_files[r : r + BATCH_SIZE]
		# batch = sorted(batch, key=lambda f: f2i[f])
		biggest_image = Image.open('{0}/{1}.jpg'.format(images_dir, batch[-1]))
		H = biggest_image.size[0]
		W = biggest_image.size[1]
		biggest_image.close()

		for file in batch[:-1]:
			im = Image.open('{0}/{1}.jpg'.format(images_dir, file))
			h = im.size[0]
			w = im.size[1]

			pixel_loss += (H - h) * W + (W - w) * h
			im.close()

	print('Average pixel loss per image: {0}'.format(pixel_loss / float(BATCH_SIZE * NUM_BATCHES)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path')
	parser.add_argument('index_file')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.index_file)
