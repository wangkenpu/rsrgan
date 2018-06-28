#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Redundant Convolutional Encoder Decoder (R-CED)
A fully convolutional neural network for speech enhancement(https://arxiv.org/pdf/1609.07132).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from tensorflow.contrib.layers import batch_norm, fully_connected

class RCED(object):

    def __init__(self, rced):
        self.rced = rced

    def __call__(self, inputs, labels, reuse=False):
        """Build CNN models. On first pass will make vars."""
        self.inputs = inputs
        self.labels = labels

        outputs = self.infer(reuse)

        return outputs

    def infer(self, reuse):
        rced = self.rced
        activation_fn = tf.nn.relu
        is_training = True

        input_dim = rced.input_dim
        left_context = rced.left_context
        right_context = rced.right_context
        splice_dim = left_context + 1 + right_context

        in_dims = self.inputs.get_shape().as_list()
        if len(in_dims) == 2:
            # shape format [batch, width]
            dims = self.inputs.get_shape().as_list()
            assert dims[0] == rced.batch_size
            inputs = tf.reshape(self.inputs, [dims[0], splice_dim, input_dim])
            inputs = tf.expand_dims(inputs, -1)
        elif len(in_dims) == 3:
            # shape format [batch, length, width]
            dims = self.inputs.get_shape().as_list()
            assert dims[0] == 1
            inputs = tf.squeeze(self.inputs, [0])
            inputs = tf.reshape(self.inputs, [-1, splice_dim, input_dim])
            inputs = tf.expand_dims(inputs, -1)

        # If test of cv , BN should use global mean / stddev
        if rced.cross_validation:
            is_training = False

        with tf.variable_scope('g_model') as scope:
            if reuse:
                scope.reuse_variables()

            if rced.batch_norm:
                normalizer_fn = batch_norm
                normalizer_params = {
                    "is_training": is_training,
                    "scale": True,
                    "renorm": True
                }
            else:
                normalizer_fn = None
                normalizer_params = None

            if rced.l2_scale > 0.0 and is_training:
                weights_regularizer = l2_regularizer(rced.l2_scale)
            else:
                weights_regularizer = None
                keep_prob = 1.0

            if not reuse:
                print("*** Generator summary ***")
                print("G inputs shape: {}".format(inputs.get_shape()))

            # inputs format [batch, in_height, in_width, in_channels]
            # filters format [filter_height, filter_width, in_channels, out_channels]
            filters_num = [12, 16, 20, 24, 32, 24, 20, 16, 12]
            filters_width = [13, 11, 9, 7, 7, 7, 9, 11, 13]
            assert len(filters_num) == len(filters_num)
            for i in range(len(filters_num)):
                inputs = tf.contrib.layers.conv2d(inputs, filters_num[i],
                        [splice_dim, filters_width[i]],
                        activation_fn=activation_fn,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_initializer=xavier_initializer(),
                        weights_regularizer=weights_regularizer,
                        biases_initializer=tf.zeros_initializer())
                if not reuse:
                    print("Conv{} layer output shape: {}".format(
                        i+1, inputs.get_shape()), end=" *** ")
                    self.nnet_info(normalizer_fn, rced.keep_prob, weights_regularizer)

            # Linear output
            # inputs = tf.reshape(inputs, [rced.batch_size, -1])
            inputs = tf.reshape(inputs, [-1, splice_dim * input_dim * filters_num[-1]])
            y = fully_connected(inputs, rced.output_dim,
                                activation_fn=None,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.constant_initializer(0.1))
            if not reuse:
                print("G output shape: {}".format(y.get_shape()))
                sys.stdout.flush()

        return y

    def nnet_info(self, batch_norm, keep_prob, weights_regularizer):
        if batch_norm is not None:
            print("use batch normalization", end=" *** ")
        if keep_prob != 1.0:
            print("keep prob is {}".format(keep_prob),
                  end=" *** ")
        if weights_regularizer is not None:
            print("L2 regularizer scale is {}".format(self.rced.l2_scale),
                  end=" *** ")

        print()
