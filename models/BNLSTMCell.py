#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""LSTM cell with Recurrent Batch Normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.layers import xavier_initializer


# Thanks to https://github.com/OlavHN/bnlstm
def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.999):
  """Batch Normalization."""
  with tf.variable_scope(name_scope):
    size = inputs.get_shape().as_list()[1]

    scale = tf.get_variable(
        'scale', [size], initializer=tf.constant_initializer(0.1))
    offset = tf.get_variable(
        'offset', [size], initializer=tf.constant_initializer(0.0))

    moving_mean = tf.get_variable('moving_mean', [size],
        initializer=tf.zeros_initializer(), trainable=False)
    moving_var = tf.get_variable('moving_var', [size],
        initializer=tf.ones_initializer(), trainable=False)
    batch_mean, batch_var = tf.nn.moments(inputs, [0])

    # The following part is based on the implementation of :
    # https://github.com/cooijmanstim/recurrent-batch-normalization
    train_mean_op = tf.assign(moving_mean,
        moving_mean * decay + batch_mean * (1 - decay))
    train_var_op = tf.assign(moving_var,
        moving_var * decay + batch_var * (1 - decay))

    if is_training is True:
      with tf.control_dependencies([train_mean_op, train_var_op]):
        return tf.nn.batch_normalization(
            inputs, batch_mean, batch_var, offset, scale, epsilon)
    else:
      return tf.nn.batch_normalization(
          inputs, moving_mean, moving_var, offset, scale, epsilon)


class BNLSTMCell(RNNCell):
  """LSTM cell with Recurrent Batch Normalization.

  This implementation is based on:
    http://arxiv.org/abs/1603.09025
    https://github.com/tam17aki/recurrent-batchnorm-tensorflow/blob/master/BN_LSTMCell.py
  """
  def __init__(self, num_units,
               is_training=True,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=tf.tanh, reuse=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      is_training: bool, set True when training.
      use_peepholes: bool, set True to enable diagonal/peephole
        connections.
      cell_clip: (optional) A float value, if provided the cell state
        is clipped by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight
        matrices.
      num_proj: (optional) int, The output dimensionality for
        the projection matrices.  If None, no projection is performed.
      forget_bias: Biases of the forget gate are initialized by default
        to 1 in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.
      activation: Activation function of the inner states. Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(BNLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      tf.logging.log_first_n(
        tf.logging.WARN,
        "%s: Using a concatenated state is slower and "
        " will soon be deprecated.  Use state_is_tuple=True.", 1, self)

    self.num_units = num_units
    self.is_training = is_training
    self.use_peepholes = use_peepholes
    self.cell_clip = cell_clip
    self.num_proj = num_proj
    self.proj_clip = proj_clip
    self.initializer = initializer
    self.forget_bias = forget_bias
    self.state_is_tuple = state_is_tuple
    self.activation = activation

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
   Returns:
     A tuple containing:
     - A `2-D, [batch x output_dim]`, Tensor representing the output of the
       LSTM after reading `inputs` when previous state was `state`.
       Here output_dim is:
          num_proj if num_proj was set,
          num_units otherwise.
     - Tensor(s) representing the new state of LSTM after reading `inputs` when
       the previous state was `state`.  Same type and shape(s) as `state`.
    """

    num_proj = self.num_units if self.num_proj is None else self.num_proj

    if self.state_is_tuple:
      (c_prev, h_prev) = state
    else:
      c_prev = tf.slice(state, [0, 0], [-1, self.num_units])
      h_prev = tf.slice(state, [0, self.num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]

    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope or type(self).__name__):

      W_xh = tf.get_variable(
          'input_kernel',
          [input_size, 4 * self.num_units],
          initializer=self.initializer)
      W_hh = tf.get_variable(
          'state_kernel',
          [num_proj, 4 * self.num_units],
          initializer=self.initializer)

      xh = tf.matmul(inputs, W_xh)
      hh = tf.matmul(h_prev, W_hh)

      bn_xh = batch_norm(xh, "input", self.is_training)
      bn_hh = batch_norm(hh, "state", self.is_training)

      bias = tf.get_variable('bias', [4 * self.num_units])
      # i:input gate, j:new input, f:forget gate, o:output gate
      lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
      i, j, f, o = tf.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

      # Diagonal connections
      if self.use_peepholes:
        w_f_diag = tf.get_variable(
            "W_F_diag", shape=[self.num_units], dtype=dtype)
        w_i_diag = tf.get_variable(
            "W_I_diag", shape=[self.num_units], dtype=dtype)
        w_o_diag = tf.get_variable(
            "W_O_diag", shape=[self.num_units], dtype=dtype)

      if self.use_peepholes:
        c = (c_prev * tf.sigmoid(f + self.forget_bias + w_f_diag * c_prev) +
            tf.sigmoid(i + w_i_diag * c_prev) * self.activation(j))
      else:
        c = (c_prev * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) *
            self.activation(j))

      if self.cell_clip is not None:
        c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)

      bn_c = batch_norm(c, "cell", self.is_training)

      if self.use_peepholes:
          h = tf.sigmoid(o + w_o_diag * c) * self.activation(bn_c)
      else:
          h = tf.sigmoid(o) * self.activation(bn_c)

      if self.num_proj is not None:
        w_proj = tf.get_variable(
            "projection/kernel", [self.num_units, num_proj], dtype=dtype)

        h = tf.matmul(h, w_proj)
        if self.proj_clip is not None:
          h = tf.clip_by_value(h, -self.proj_clip, self.proj_clip)

    new_state = (LSTMStateTuple(c, h) if self.state_is_tuple else
                 tf.concat(values=[c, h], axis=1))
    return h, new_state
