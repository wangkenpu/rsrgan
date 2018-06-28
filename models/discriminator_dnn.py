#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Build the feed forward fully connected neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from utils.ops import *


def discriminator_dnn(self, inputs, reuse=False):
  """Build DNN model. On first pass will make vars."""
  units = 1024
  hidden_layers = 3
  activation_fn = tf.nn.relu
  relu_stddev = np.sqrt(2 / units)
  relu_itializer = tf.truncated_normal_initializer(mean=0.0, stddev=relu_stddev)

  # If test of cv , BN should use global mean / stddev
  is_training = False if self.cross_validation else True

  with tf.variable_scope('d_model') as scope:
    if reuse:
      scope.reuse_variables()

    if self.batch_norm:
      normalizer_fn = batch_norm
      normalizer_params = {
          "is_training": is_training,
          "scale": True,
          "renorm": True
      }
    else:
      normalizer_fn = None
      normalizer_params = None

    if self.l2_scale > 0.0 and is_training:
      weights_regularizer = l2_regularizer(self.l2_scale)
    else:
      weights_regularizer = None
      keep_prob = 1.0

    sys.stdout.flush()
    # Apply input noisy layer
    if not reuse:
      print("*** Discriminator summary ***")
      print("D inputs shape: {}".format(inputs.get_shape()))

    # inputs = gaussian_noise_layer(inputs, self.disc_noise_std)

    h = fully_connected(inputs, units,
                        activation_fn=activation_fn,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_initializer=relu_itializer,
                        weights_regularizer=weights_regularizer,
                        biases_initializer=tf.zeros_initializer())
    h = dropout(h, self.keep_prob)
    if not reuse:
      print("D layer 1 output shape: {}".format(h.get_shape()), end=" *** ")
      nnet_info(self, normalizer_fn, self.keep_prob, weights_regularizer)

    for layer in range(hidden_layers):
      h = fully_connected(h, units,
                          activation_fn=activation_fn,
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params,
                          weights_initializer=relu_itializer,
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.zeros_initializer())
      h = dropout(h, self.keep_prob)
      if not reuse:
        print("D layer {} output shape: {}".format(
            layer+2, h.get_shape()), end=" *** ")
        nnet_info(self, normalizer_fn, self.keep_prob, weights_regularizer)

    # Output layer
    y = fully_connected(h, 1,
                        activation_fn=None,
                        weights_initializer=xavier_initializer(),
                        weights_regularizer=weights_regularizer,
                        biases_initializer=tf.zeros_initializer())
    y = tf.clip_by_value(y, -0.5, 1.5)
    if not reuse:
      print("d output shape: {}".format(y.get_shape()))
      print("****************************************")
      sys.stdout.flush()
  return y

def dropout(x, keep_prob):
  if keep_prob != 1.0:
    y = tf.nn.dropout(x, keep_prob)
  else:
    y = x
  return y

def nnet_info(self, batch_norm, keep_prob, weights_regularizer):
  if batch_norm is not None:
    print("use batch normalization", end=" *** ")
  if keep_prob != 1.0:
    print("keep prob is {}".format(keep_prob), end=" *** ")
  if weights_regularizer is not None:
    print("L2 regularizer scale is {}".format(self.l2_scale), end=" *** ")
  print()
