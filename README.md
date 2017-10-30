# Deep Learning Final Project
### Re-Implementing "Fully Convolutional Networks for Semantic Segmentation"

## Utilities

* `remove_non_segmentation_images.py`

* `index_images.py`
   From project root, run `python python/index_images.py data/VOCdevkit data/images_index.txt` to generate index for sorting by size.

* `test_index.py`
   From project root, run `python python/test_index.py data/VOCdevkit data/images_index.txt` to get an estimate of the average number of pixels needed for padding per image, given the current indexing.  Current scheme probably fine - gets value down to about 540, compared to ~33,000 with random selection.
