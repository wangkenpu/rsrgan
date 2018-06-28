#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from models.discriminator_dnn import *
from models.discriminator_lstm import *
from models.lstm import *
from models.res_lstm_l import RES_LSTM_L
from models.res_lstm_base import RES_LSTM_BASE
from utils.ops import *

class Model(object):

  def __init__(self, name='BaseModel'):
    self.name = name

  def save(self, save_dir, step):
    model_name = self.name
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    if not hasattr(self, 'saver'):
      self.saver = tf.train.Saver(max_to_keep=10)
    self.saver.save(self.sess,
                    os.path.join(save_dir, model_name),
                    global_step=step)

  def load(self, save_dir, model_file=None, moving_average=False):
    if not os.path.exists(save_dir):
      print('[!] Checkpoints path does not exist...')
      return False
    print('[*] Reading checkpoints...')
    if model_file is None:
      ckpt = tf.train.get_checkpoint_state(save_dir)
      if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      else:
        return False
    else:
      ckpt_name = model_file

    if moving_average:
      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(
                                               self.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
    else:
      saver = tf.train.Saver()
    saver.restore(self.sess, os.path.join(save_dir, ckpt_name))
    print('[*] Read {}'.format(ckpt_name))
    return True


class GAN_RNN(Model):
  """Generative Adversarial Network for Speech Enhancement"""
  def __init__(self, sess, args, devices,
               inputs, labels, lengths, cross_validation=False, name='GAN_RNN'):
    super(GAN_RNN, self).__init__(name)
    self.sess = sess
    self.cross_validation = cross_validation
    self.MOVING_AVERAGE_DECAY = 0.9999
    self.max_grad_norm = 15
    if cross_validation:
      self.keep_prob = 1.0
    else:
      self.keep_prob = args.keep_prob
    self.batch_norm = args.batch_norm
    self.batch_size = args.batch_size
    self.devices = devices
    self.save_dir = args.save_dir
    if not cross_validation:
      self.writer = tf.summary.FileWriter(os.path.join(
          args.save_dir,'train'), sess.graph)
    else:
      self.writer = tf.summary.FileWriter(os.path.join(
          args.save_dir,'eval'), sess.graph)
    self.l2_scale = args.l2_scale
    # data
    self.input_dim = args.input_dim
    self.output_dim = args.output_dim
    self.left_context = args.left_context
    self.right_context = args.right_context
    self.batch_size = args.batch_size
    self.inputs = inputs
    self.labels = labels
    self.lengths = lengths
    # Batch Normalization
    self.batch_norm = args.batch_norm
    self.g_disturb_weights = False
    # Train config
    self.disc_updates = args.disc_updates
    self.gen_updates = args.gen_updates
    # G's supervised loss weight
    self.mse_lambda = tf.Variable(args.init_mse_weight, trainable=False)
    # Discriminator config
    self.d_clip_weights = False
    self.disc_noise_std = tf.Variable(args.init_disc_noise_std, trainable=False)
    # define the functions
    self.discriminator = discriminator_lstm
    self.d_learning_rate = tf.Variable(args.d_learning_rate, trainable=False)
    self.g_learning_rate = tf.Variable(args.g_learning_rate, trainable=False)

    # Use soft and noise labels
    self.d_real = tf.Variable(1.0, trainable=False)
    self.d_fake = tf.Variable(0.0, trainable=False)

    if args.g_type == 'lstm':
      self.generator = LSTM(self)
    elif args.g_type == 'res_lstm_l':
      self.generator = RES_LSTM_L(self)
    elif args.g_type == 'res_lstm_base':
      self.generator = RES_LSTM_BASE(self)
    else:
      raise ValueError('Unrecognized G type {}'.format(args.g_type))
    if labels is None:
      self.generator(inputs, labels, lengths, reuse=False)
    else:
      self.build_model(inputs, labels, lengths)

  def build_model(self, inputs, labels, lengths):
    all_d_grads = []
    all_g_grads = []
    # d_opt = tf.train.RMSPropOptimizer(self.d_learning_rate)
    # g_opt = tf.train.RMSPropOptimizer(self.g_learning_rate)
    d_opt = tf.train.GradientDescentOptimizer(self.d_learning_rate)
    # g_opt = tf.train.GradientDescentOptimizer(self.g_learning_rate)
    # d_opt = tf.train.AdamOptimizer(self.d_learning_rate)
    g_opt = tf.train.AdamOptimizer(self.g_learning_rate)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        self.MOVING_AVERAGE_DECAY)

    with tf.variable_scope(tf.get_variable_scope()):
      for idx, device in enumerate(self.devices):
        with tf.device("/%s" % device):
          with tf.name_scope("device_%s" % idx):
            with variables_on_gpu():
              self.build_model_single_gpu(idx, inputs, labels, lengths)
              tf.get_variable_scope().reuse_variables()
              if not self.cross_validation:
                g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                 ".*g_model")
                d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                 ".*d_model")
                with tf.control_dependencies(d_update_ops):
                  d_grads = d_opt.compute_gradients(self.d_losses[-1],
                                                    var_list=self.d_vars)
                  all_d_grads.append(d_grads)
                with tf.control_dependencies(g_update_ops):
                  g_grads = g_opt.compute_gradients(self.g_losses[-1],
                                                    var_list=self.g_vars)
                  all_g_grads.append(g_grads)
    if not self.cross_validation:
      avg_d_grads = average_gradients(all_d_grads)
      for i, (g, v) in enumerate(avg_d_grads):
        avg_d_grads[i] = (tf.clip_by_norm(g, self.max_grad_norm), v)
      avg_g_grads = average_gradients(all_g_grads)
      for i, (g, v) in enumerate(avg_g_grads):
        avg_g_grads[i] = (tf.clip_by_norm(g, self.max_grad_norm), v)
      d_apply_gradient_op = d_opt.apply_gradients(avg_d_grads)
      g_apply_gradient_op = g_opt.apply_gradients(avg_g_grads)
      d_variables_averages_op = variable_averages.apply(self.d_vars)
      g_variables_averages_op = variable_averages.apply(self.g_vars)
      # Group all updates to into a single train op.
      self.d_opt = tf.group(d_apply_gradient_op, d_variables_averages_op)
      self.g_opt = tf.group(g_apply_gradient_op, g_variables_averages_op)

  def build_model_single_gpu(self, gpu_idx, inputs, labels, lengths):
    start_dim = self.input_dim * self.left_context
    # d_inputs = tf.slice(inputs, [0, 0, start_dim],
    #                     [-1, -1, self.input_dim], name='d_inputs')
    if gpu_idx == 0:
      g = self.generator(inputs, labels, lengths, reuse=False)
      # Make a dummy copy of discriminator to have variables and then
      # be able to set up the variable reuse for all other devices
      # merge along channels and this would be a real batch
      # dummy_joint = tf.concat([d_inputs, labels], -1)
      dummy_joint = labels
      dummy = self.discriminator(self, dummy_joint, lengths, reuse=False)

    g = self.generator(inputs, labels, lengths, reuse=True)

    # d_rl_joint = tf.concat([d_inputs, labels], -1)
    # d_fk_joint = tf.concat([d_inputs, g], -1)
    d_rl_joint = labels
    d_fk_joint = g

    # Build d loss
    d_rl_logits = self.discriminator(self, d_rl_joint, lengths, reuse=True)
    d_fk_logits = self.discriminator(self, d_fk_joint, lengths, reuse=True)

    if gpu_idx == 0:
      self.get_vars()

    # Make disc variables summaries
    self.d_rl_summ = histogram_summary("d_real", d_rl_logits)
    self.d_fk_summ = histogram_summary("d_fake", d_fk_logits)
    self.real_clean_summ = histogram_summary('real_clean', inputs)
    self.real_noise_summ = histogram_summary('real_noise', labels)
    self.gen_summ = histogram_summary('g_clean', g)

    if gpu_idx == 0:
      self.g_losses = []
      self.g_mse_losses = []
      self.g_adv_losses = []
      self.g_l2_losses = []
      self.d_rl_losses = []
      self.d_fk_losses = []
      self.d_losses = []


    # real_logits = d_rl_logits
    # fake_logits = d_fk_logits
    # d_class_loss = softmax_cross_entropy_with_logits(real_logits, fake_logits)
    # d_rl_loss = tf.reduce_mean(
    #     tf.squared_difference(real_logits[:,:,0:1], self.d_real))
    # d_fk_loss = tf.reduce_mean(
    #     tf.squared_difference(fake_logits[:,:,1:2], self.d_fake))
    # g_adv_loss = tf.reduce_mean(
    #     tf.squared_difference(fake_logits[:,:,1:2], self.d_real))
    d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, self.d_real))
    d_fk_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, self.d_fake))
    g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, self.d_real))

    # d_loss = d_rl_loss + d_fk_loss + d_class_loss
    d_loss = d_rl_loss + d_fk_loss

    # Add MSE loss to G
    g_mse_loss = 0.5 * tf.losses.mean_squared_error(g, labels) * self.output_dim
    if not self.cross_validation and self.l2_scale > 0.0:
      tvars = [v for v in self.g_vars if "bias" not in v.name]
      reg_losses = tf.reduce_sum([tf.nn.l2_loss(v) for v in tvars])
      g_l2_loss = reg_losses * self.l2_scale
    else:
      g_l2_loss = tf.constant(0.0)

    g_loss = g_adv_loss + self.mse_lambda * g_mse_loss + g_l2_loss

    self.d_rl_losses.append(d_rl_loss)
    self.d_fk_losses.append(d_fk_loss)
    self.d_losses.append(d_loss)
    self.g_adv_losses.append(g_adv_loss)
    self.g_mse_losses.append(g_mse_loss)
    self.g_l2_losses.append(g_l2_loss)
    self.g_losses.append(g_loss)

    self.d_rl_loss_summ = scalar_summary("d_rl_loss",
                                         tf.reduce_mean(self.d_rl_losses))
    self.d_fk_loss_summ = scalar_summary("d_fk_loss",
                                         tf.reduce_mean(self.d_fk_losses))
    self.d_loss_summ = scalar_summary("d_loss",
                                      tf.reduce_mean(self.d_losses))
    self.g_adv_loss_summ = scalar_summary("g_adv_loss",
                                          tf.reduce_mean(self.g_adv_losses))
    self.g_mse_loss_summ = scalar_summary("g_mse_loss",
                                          tf.reduce_mean(self.g_mse_losses))
    self.g_l2_loss_summ = scalar_summary("g_l2_loss",
                                         tf.reduce_mean(self.g_l2_losses))
    self.g_loss_summ = scalar_summary("g_loss",
                                      tf.reduce_mean(self.g_losses))

    summaries = [self.d_rl_loss_summ,
                 self.d_fk_loss_summ,
                 self.d_loss_summ,
                 self.g_adv_loss_summ,
                 self.g_mse_loss_summ,
                 self.g_l2_loss_summ,
                 self.g_loss_summ,
                 self.d_rl_summ,
                 self.d_fk_summ,
                 self.real_clean_summ,
                 self.real_noise_summ,
                 self.gen_summ]

    self.summaries = tf.summary.merge(summaries)


  def get_vars(self):
    t_vars = tf.trainable_variables()
    self.d_vars_dict = {}
    self.g_vars_dict = {}
    for var in t_vars:
      if var.name.startswith('d_'):
        self.d_vars_dict[var.name] = var
      if var.name.startswith('g_'):
        self.g_vars_dict[var.name] = var
    self.d_vars = self.d_vars_dict.values()
    self.g_vars = self.g_vars_dict.values()
    for x in self.d_vars:
      assert x not in self.g_vars
    for x in self.g_vars:
      assert x not in self.d_vars
    for x in t_vars:
      assert x in self.g_vars or x in self.d_vars, x.name
    self.all_vars = t_vars
    if self.d_clip_weights and not self.cross_validation:
      print('Clipping D weights')
      sys.stdout.flush()
      self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars]
    else:
      print('Not clipping D weights')
      sys.stdout.flush()
    if self.g_disturb_weights and not self.cross_validation:
      stddev = 0.00001
      print("Add Gaussian noise to G weights (stddev = %s)" % (stddev))
      sys.stdout.flush()
      self.g_disturb = [v.assign(
          tf.add(v, tf.truncated_normal([], 0, stddev))) for v in self.g_vars]
    else:
      print("Not add noise to G weights")
