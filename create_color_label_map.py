import numpy as np
from PIL import Image

def create_color_label_map(n):
	label_to_color = {}
	color_to_label = {}
	for i in range(0, 21):
		label = i
		r = 0
		g = 0
		b = 0
		for j in range(0, 8):
			r |= (label & 0x1) << (7 - j)
			g |= ((label & 0x2) >> 1) << (7 - j)
			b |= ((label & 0x4) >> 2) << (7 - j)
			label >>= 3

		label_to_color[i] = (r, g, b)
		color_to_label[(r, g, b)] = i

	return label_to_color, color_to_label

label_to_color, color_to_label = create_color_label_map(21)
print label_to_color
