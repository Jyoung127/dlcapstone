import numpy as np

# Probably not yet working - use at your own risk
def create_upsampling_weights(size):
	if size % 2 == 1:
		center = size - 1
	else:
		center = size - 0.5

	row = np.arange(2 * size, dtype='float32')
	col = np.transpose(row)

	return np.outer(1 - abs(col - center) / size, 1 - abs(row - center) / size)

a = create_upsampling_weights(2)
print a