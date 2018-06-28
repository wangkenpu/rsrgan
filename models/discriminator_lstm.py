#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Build the feed forward fully connected neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

sys.path.append(os.path.dirname(sys.path[0]))
from utils.ops import *


def discriminator_lstm(self, inputs, lengths, reuse=False):
  """Build DNN model. On first pass will make vars."""
  lstm_cell_size = 256
  num_projection = 40
  lstm_num_layer = 2

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

    inputs = gaussian_noise_layer(inputs, self.disc_noise_std)

    # h = fully_connected(inputs, num_projection,
    #                     activation_fn=leakyrelu,
    #                     normalizer_fn=normalizer_fn,
    #                     normalizer_params=normalizer_params,
    #                     weights_initializer=xavier_initializer(),
    #                     weights_regularizer=weights_regularizer,
    #                     biases_initializer=tf.zeros_initializer())

    def lstm_cell():
      return tf.contrib.rnn.LSTMCell(lstm_cell_size, use_peepholes=True,
                                     initializer=xavier_initializer(),
                                     num_proj=num_projection,
                                     forget_bias=1.0, state_is_tuple=True,
                                     activation=tf.tanh,
                                     reuse=reuse)
    attn_cell = lstm_cell
    if is_training and self.keep_prob < 1.0:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=self.keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(lstm_num_layer)], state_is_tuple=True)

    initial_states = cell.zero_state(self.batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                        sequence_length=lengths,
                                        initial_state=initial_states,
                                        dtype=tf.float32,
                                        time_major=False)

    if not reuse:
      print("D hidden layer number is {}".format(lstm_num_layer))
      print("D cell size is {}".format(lstm_cell_size))
      print("D projection num is {}".format(num_projection))
    sys.stdout.flush()

    # Output layer
    y = fully_connected(outputs, 1,
                        activation_fn=None,
                        weights_initializer=xavier_initializer(),
                        weights_regularizer=weights_regularizer,
                        biases_initializer=tf.zeros_initializer())
    # y = tf.clip_by_value(y, -0.5, 1.5)
    if not reuse:
      print("d output shape: {}".format(y.get_shape()))
      print("****************************************")
      sys.stdout.flush()
  return y
