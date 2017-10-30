import os
import argparse
from PIL import Image


ALL_IMAGES_FILE_REL = 'VOC2012/ImageSets/Segmentation/trainval.txt'
IMAGES_DIR_REL = 'VOC2012/JPEGImages'


def main(voc_devkit_path, output_path):
	all_images_file = '{0}/{1}'.format(voc_devkit_path, ALL_IMAGES_FILE_REL)
	images_dir = '{0}/{1}'.format(voc_devkit_path, IMAGES_DIR_REL)

	file2size = {}
	trimmed_files = []

	files = os.listdir(images_dir)
	for file in files:
		im = Image.open('{0}/{1}'.format(images_dir, file))
		file2size[file] = im.size
		trimmed_files.append(file.split('.')[0])
		im.close()

	sorted_files = sorted(trimmed_files,
		key=lambda f: file2size[f + '.jpg'][0] + file2size[f + '.jpg'][1] * 0.1)

	with open(output_path, 'w') as f_out:
		for file in sorted_files:
			print(file, file2size[file + '.jpg'])
			f_out.write(file + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('voc_devkit_path')
	parser.add_argument('output_path')
	args = parser.parse_args()
	main(args.voc_devkit_path, args.output_path)
