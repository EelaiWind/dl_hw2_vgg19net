from __future__ import print_function

import numpy as np
import cv2

IMAGE_SIZE = 224
RED_MEAN = 123.68
GREEN_MEAN = 116.779
BLUE_MEAN = 103.939
def load_image(image_path):
    image = cv2.imread(image_path)
    shorter_dimension = np.min(image.shape[:2])
    width_offet = (image.shape[0] - shorter_dimension)/2
    height_offet = (image.shape[1] - shorter_dimension)/2
    image = image[width_offet:width_offet+shorter_dimension, height_offet:height_offet+shorter_dimension]
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float)
    #opencv read image as BGR format
    image[:, :, 0] = image[:, :, 0] - BLUE_MEAN
    image[:, :, 1] = image[:, :, 1] - GREEN_MEAN
    image[:, :, 2] = image[:, :, 2] - RED_MEAN
    return image

def print_top_5(probability, label_file_path):
    with open(label_file_path, 'r') as label_file:
        labels = [ line.strip() for line in label_file.readlines()]

    for i in xrange(len(probability)):
        row_output = probability[i,:]
        sort_index = np.argsort(row_output)[::-1]
        for i in xrange(5):
            index = sort_index[i]
            print ('top %d: %s (%0.2f)' % (i+1, labels[index], row_output[index]))