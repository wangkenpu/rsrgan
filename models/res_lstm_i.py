#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Build the LSTM neural networks.
This module provides an example of definiting compute graph with tensorflow.
"""

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
from utils.ops import leakyrelu

class RES_LSTM_I(object):

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
        num_projection = 257
        lstm_num_layer = 3
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
            # h = fully_connected(inputs, num_projection,
            #                     activation_fn=leakyrelu,
            #                     normalizer_fn=normalizer_fn,
            #                     normalizer_params=normalizer_params,
            #                     weights_initializer=xavier_initializer(),
            #                     biases_initializer=tf.zeros_initializer())

            def lstm_cell():
                return tf.contrib.rnn.LSTMCell(
                    lstm_cell_size, use_peepholes=True,
                    initializer=xavier_initializer(),
                    num_proj=num_projection,
                    forget_bias=1.0, state_is_tuple=True,
                    activation=tf.tanh,
                    reuse=reuse)
            attn_cell = lstm_cell

            if is_training and lstm.keep_prob < 1.0:
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                        lstm_cell(), output_keep_prob=lstm.keep_prob)

            with tf.variable_scope("lstm_cell_1"):
                cell1 = attn_cell()
                initial_states = cell1.zero_state(lstm.batch_size, tf.float32)
                outputs1, states1 = tf.nn.dynamic_rnn(cell1, self.inputs,
                                              sequence_length=self.lengths,
                                              initial_state=initial_states,
                                              dtype=tf.float32,
                                              time_major=False)

            with tf.variable_scope("lstm_cell_2"):
                inputs2 = outputs1 + self.inputs
                cell2 = attn_cell()
                initial_states = cell2.zero_state(lstm.batch_size, tf.float32)
                outputs2, states2 = tf.nn.dynamic_rnn(cell2, inputs2,
                                              sequence_length=self.lengths,
                                              initial_state=initial_states,
                                              dtype=tf.float32,
                                              time_major=False)

            # with tf.variable_scope("lstm_cell_3"):
            #     inputs3 = outputs2 + self.inputs
            #     cell3 = attn_cell()
            #     initial_states = cell3.zero_state(lstm.batch_size, tf.float32)
            #     outputs3, states3 = tf.nn.dynamic_rnn(cell3, inputs3,
            #                                   sequence_length=self.lengths,
            #                                   initial_state=initial_states,
            #                                   dtype=tf.float32,
            #                                   time_major=False)

            # with tf.variable_scope("lstm_cell_4"):
            #     inputs4 = outputs3 + self.inputs
            #     cell4 = attn_cell()
            #     initial_states = cell4.zero_state(lstm.batch_size, tf.float32)
            #     outputs4, states4 = tf.nn.dynamic_rnn(cell4, inputs4,
            #                                   sequence_length=self.lengths,
            #                                   initial_state=initial_states,
            #                                   dtype=tf.float32,
            #                                   time_major=False)

            # with tf.variable_scope("lstm_cell_5"):
            #     inputs5 = outputs4 + self.inputs
            #     cell5 = attn_cell()
            #     initial_states = cell5.zero_state(lstm.batch_size, tf.float32)
            #     outputs5, states5 = tf.nn.dynamic_rnn(cell5, inputs5,
            #                                   sequence_length=self.lengths,
            #                                   initial_state=initial_states,
            #                                   dtype=tf.float32,
            #                                   time_major=False)

            # with tf.variable_scope("lstm_cell_6"):
            #     inputs6 = outputs5 + self.inputs
            #     cell6 = attn_cell()
            #     initial_states = cell6.zero_state(lstm.batch_size, tf.float32)
            #     outputs6, states6 = tf.nn.dynamic_rnn(cell6, inputs6,
            #                                   sequence_length=self.lengths,
            #                                   initial_state=initial_states,
            #                                   dtype=tf.float32,
            #                                   time_major=False)

            # with tf.variable_scope("lstm_cell_7"):
            #     inputs7 = outputs6 + self.inputs
            #     cell7 = attn_cell()
            #     initial_states = cell7.zero_state(lstm.batch_size, tf.float32)
            #     outputs7, states7 = tf.nn.dynamic_rnn(cell7, inputs7,
            #                                   sequence_length=self.lengths,
            #                                   initial_state=initial_states,
            #                                   dtype=tf.float32,
            #                                   time_major=False)

            # with tf.variable_scope("lstm_cell_8"):
            #     inputs8 = outputs7 + self.inputs
            #     cell8 = attn_cell()
            #     initial_states = cell8.zero_state(lstm.batch_size, tf.float32)
            #     outputs8, states8 = tf.nn.dynamic_rnn(cell8, inputs8,
            #                                   sequence_length=self.lengths,
            #                                   initial_state=initial_states,
            #                                   dtype=tf.float32,
            #                                   time_major=False)

            if not reuse:
                print("G hidden layer number is {}".format(lstm_num_layer))
                print("G cell size is {}".format(lstm_cell_size))
                print("G projection num is {}".format(num_projection))
            sys.stdout.flush()

            # Linear output
            with tf.variable_scope("forward_out"):
                # inputs9 = outputs8 + self.inputs
                # inputs9 = outputs4 + self.inputs
                inputs9 = outputs2 + self.inputs
                y = fully_connected(inputs9, lstm.output_dim,
                                    activation_fn=None,
                                    weights_initializer=xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer())
            if not reuse:
                print("G output shape: {}".format(y.get_shape()))
                sys.stdout.flush()

        return y
