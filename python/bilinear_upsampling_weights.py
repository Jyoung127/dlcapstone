import numpy as np

def bilinear_filter(size):
	if size % 2 == 0:
		center = size / 2.0 - 0.5
		width = size / 2.0
	else:
		center = size / 2.0 - 0.5
		width = size / 2.0 + 0.5

	points = np.arange(size, dtype='float32')

	return np.outer(1 - abs(points - center) / width, 1 - abs(points - center) / width)

def create_initial_bilinear_weights(size):
	W = np.zeros([size, size, 21, 21])
	bilinear_filt = bilinear_filter(size)

	for i in range(21):
		W[:,:,i,i] = bilinear_filt

	return W
