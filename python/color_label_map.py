import numpy as np
from PIL import Image

MANUAL_VOID_RGB = (1, 1, 1)
MANUAL_VOID_LABEL = 254

# Creates the color label map used in the PASCAL VOC 2012 image segmentation labels.
# Class 0: Background
# Class 1-20: 
# Class 255: Border / ambiguous pixels (ignore while scoring)
def create_color_label_map():
	label_to_color = {}
	color_to_label = {}
	for i in range(0, 256):
		label = i
		r = 0
		g = 0
		b = 0
		for j in range(0, 8):
			r |= (label & 0x1) << (7 - j)
			g |= ((label & 0x2) >> 1) << (7 - j)
			b |= ((label & 0x4) >> 2) << (7 - j)
			label >>= 3

		label_to_color[i] = [r, g, b]
		color_to_label[(r, g, b)] = i

	label_to_color[MANUAL_VOID_LABEL] = MANUAL_VOID_RGB
	color_to_label[MANUAL_VOID_RGB] = MANUAL_VOID_LABEL
	return label_to_color, color_to_label

def label_image_to_rgb(image, label_to_color_map):
	(width, height) = image.shape
	return np.apply_along_axis(lambda x: label_to_color_map[x[0]], 2, np.reshape(image, (width, height, 1)))

def rgb_image_to_label(image, color_to_label_map):
	return np.apply_along_axis(lambda x: color_to_label_map[(x[0], x[1], x[2])], 2, image)
	