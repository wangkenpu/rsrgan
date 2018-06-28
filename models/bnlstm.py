#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Build the LSTM neural networks.
This module provides an example of definiting compute graph with tensorflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

from models.BNLSTMCell import *
from tensorflow.contrib.layers import batch_norm, fully_connected
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

class BNLSTM(object):

    def __init__(self, lstm):
        self.lstm = lstm

    def __call__(self, inputs, labels, lengths, reuse=False):
        """Build LSTM model. On first pass will make vars."""
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths

        outputs = self.infer(reuse)

        return outputs

    def infer(self, reuse):
        lstm = self.lstm
        lstm_cell_size = 760
        num_projection = 280
        lstm_num_layer = 3
        relu_stddev = np.sqrt(2 / num_projection)
        relu_itializer = tf.truncated_normal_initializer(mean=0.0,
                                                         stddev=relu_stddev)
        in_dims = self.inputs.get_shape().as_list()
        assert len(in_dims) == 3

        if lstm.cross_validation:
            is_training = False
        else:
            is_training = True

        with tf.variable_scope("g_model") as scope:
            if reuse:
                scope.reuse_variables()

            if lstm.batch_norm:
                normalizer_fn = batch_norm
                normalizer_params = {
                    "is_training": is_training,
                    "scale": True,
                    "renorm": True
                }
            else:
                normalizer_fn = None
                normalizer_params = None

            if not is_training:
                lstm.keep_prob = 1.0

            if not reuse:
                print("****************************************")
                print("*** Generator summary ***")
                print("G inputs shape: {}".format(self.inputs.get_shape()))
            sys.stdout.flush()

            inputs = self.inputs
            h = fully_connected(inputs, num_projection,
                                activation_fn=tf.nn.relu,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                weights_initializer=relu_itializer,
                                biases_initializer=tf.zeros_initializer())

            def lstm_cell():
                return BNLSTMCell(lstm_cell_size,
                                  is_training=is_training,
                                  use_peepholes=True,
                                  initializer=xavier_initializer(),
                                  num_proj=num_projection,
                                  forget_bias=1.0,
                                  state_is_tuple=True,
                                  activation=tf.tanh)
            attn_cell = lstm_cell

            if is_training and lstm.keep_prob < 1.0:
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        lstm_cell(), output_keep_prob=lstm.keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(lstm_num_layer)], state_is_tuple=True)

            initial_states = cell.zero_state(lstm.batch_size, tf.float32)
            outputs, states = tf.nn.dynamic_rnn(cell, h,
                                                sequence_length=self.lengths,
                                                initial_state=initial_states,
                                                dtype=tf.float32,
                                                time_major=False)

            if not reuse:
                print("G hidden layer number is {}".format(lstm_num_layer))
                print("G cell size is {}".format(lstm_cell_size))
                print("G projection num is {}".format(num_projection))
            sys.stdout.flush()

            # Linear output
            y = fully_connected(outputs, lstm.output_dim,
                                activation_fn=None,
                                weights_initializer=xavier_initializer(),
                                biases_initializer=tf.zeros_initializer())
            if not reuse:
                print("G output shape: {}".format(y.get_shape()))
                sys.stdout.flush()

        return y
