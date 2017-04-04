
from __future__ import print_function

import sys
import os

import utils
from vgg19_model import Vgg19Model

import tensorflow as tf
import numpy as np
import cv2

if len(sys.argv) < 2:
    print('usage: %s <image file>' % sys.argv[0], file=sys.stderr)
    sys.exit()

model = Vgg19Model(os.path.join('..', 'vggnet', 'vgg19.npy'))
input_tensor = tf.placeholder("float", shape=[None, utils.IMAGE_SIZE, utils.IMAGE_SIZE, 3])
output_tensor = model.build_model(input_tensor)

image = utils.load_image(sys.argv[1])
image = image.reshape([1, utils.IMAGE_SIZE, utils.IMAGE_SIZE, 3])
cv2.imwrite('tmp.jpg', image[0])

for tensor in tf.trainable_variables():
    print(tensor.name)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    output = sess.run(output_tensor, feed_dict={input_tensor:image})
    utils.print_top_5(output, os.path.join('..', 'vggnet', 'image_labels.txt'))
