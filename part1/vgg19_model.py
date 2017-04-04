from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

KERNEL_SIZE=3

class Vgg19Model:
    def __init__(self, parameter_dict_path):
        self.m_parameter_dict=np.load(parameter_dict_path, encoding='latin1').item()
        print('load pre-trained parameter from %s' % parameter_dict_path)


    def _load_layer_parameter(self, name):
        if name in self.m_parameter_dict:
            parameter = self.m_parameter_dict[name]
            weight =  tf.constant(parameter[0], name='weight')
            bias =  tf.constant(parameter[1], name='bias')
            return weight, bias
        else:
            print('Cannot find pre-trained parameter "%s"' % name, file=sys.stderr)
            sys.exit()

    def _build_convolution_layer(self, layer_name, input_tensor):
        with tf.variable_scope(layer_name):
            weight, bias = self._load_layer_parameter(layer_name)
            convolution = tf.nn.conv2d(input_tensor, weight, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(convolution, bias)
            relu = tf.nn.relu(pre_activation)
            return relu

    def _build_fully_connect_layer(self, layer_name, input_tensor):
        if tf.rank(input_tensor) != 2:
            dimension_product= 1
            for dim in input_tensor.get_shape().as_list()[1:]:
                dimension_product *= dim
            input_tensor = tf.reshape(input_tensor, [-1, dimension_product])
        with tf.variable_scope(layer_name):
           weight, bias = self._load_layer_parameter(layer_name)
           return tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)


    def _build_convolution_block(self, block_prefix, input_tensor, stack):
        assert stack > 0
        tensor = input_tensor
        for i in xrange(1, stack+1):
            tensor = self._build_convolution_layer('%s_%d' % (block_prefix, i), tensor)
        return tensor

    def _build_max_pool_layer(self, input_tensor):
        output = tf.nn.max_pool(input_tensor, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(output.get_shape().as_list())
        return output

    def build_model(self, input_tensor):
        conv_1 = self._build_convolution_block('conv1', input_tensor, 2)
        pool_1 = self._build_max_pool_layer(conv_1)
        conv_2 = self._build_convolution_block('conv2', pool_1, 2)
        pool_2 = self._build_max_pool_layer(conv_2)
        conv_3 = self._build_convolution_block('conv3', pool_2, 4)
        pool_3 = self._build_max_pool_layer(conv_3)
        conv_4 = self._build_convolution_block('conv4', pool_3, 4)
        pool_4 = self._build_max_pool_layer(conv_4)
        conv_5 = self._build_convolution_block('conv5', pool_4, 4)
        pool_5 = self._build_max_pool_layer(conv_5)

        fc_6 = self._build_fully_connect_layer('fc6', pool_5)
        relu_6 = tf.nn.relu(fc_6)
        fc_7 = self._build_fully_connect_layer('fc7', relu_6)
        relu_7 = tf.nn.relu(fc_7)
        fc_8 = self._build_fully_connect_layer('fc8', relu_7)
        softmax = tf.nn.softmax(fc_8)
        return softmax
