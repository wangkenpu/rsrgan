#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from models.discriminator import *
from models.generator import *
from utils.bnorm import VBN
from utils.ops import *


class Model(object):

    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_dir, step):
        model_name = self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess,
                        os.path.join(save_dir, model_name),
                        global_step=step)

    def load(self, save_dir, model_file=None):
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
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_dir, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True


class SEGAN(Model):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, sess, args, devices,
                 inputs, labels, cross_validation=False, name='SEGAN'):
        super(SEGAN, self).__init__(name)
        self.sess = sess
        self.cross_validation = cross_validation
        self.keep_prob = 1.0
        if cross_validation:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = 0.5
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.devices = devices
        # type of deconv
        self.deconv_type = args.deconv_type
        # specify if use biases or not
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        # clip D values
        self.d_clip_weights = False
        # apply VBN or regular BN
        self.disable_vbn = False
        self.save_dir = args.save_dir
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = 1
        # dilation factors per layer (only in atrous conv G config)
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # num fmaps for AutoEncoder SEGAN (v1)
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        # Define D fmaps
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.init_noise_std = args.init_noise_std
        self.disc_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        self.disc_noise_std_sum = scalar_summary('disc_noise_std',
                                                  self.disc_noise_std)
        # data
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.batch_size = args.batch_size
        # G's supervised loss weight
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        # define the functions
        self.discriminator = discriminator
        # register G non linearity
        self.g_nl = args.g_nl
        self.d_learning_rate = args.d_learning_rate
        self.g_learning_rate = args.g_learning_rate
        if args.g_type == 'ae':
            self.generator = AEGenerator(self)
        elif args.g_type == 'dfeat':
            self.generator = Generator(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(args.g_type))
        self.build_model(inputs, labels)

    def build_model(self, inputs, labels):
        all_d_grads = []
        all_g_grads = []
        d_opt = tf.train.RMSPropOptimizer(self.d_learning_rate)
        g_opt = tf.train.RMSPropOptimizer(self.g_learning_rate)
        # d_opt = tf.train.GradientDescentOptimizer(config.d_learning_rate)
        # g_opt = tf.train.GradientDescentOptimizer(config.g_learning_rate)
        # d_opt = tf.train.AdamOptimizer(config.d_learning_rate,
        #                                beta1=config.beta_1)
        # g_opt = tf.train.AdamOptimizer(config.g_learning_rate,
        #                                beta1=config.beta_1)

        with tf.variable_scope(tf.get_variable_scope()):
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_gpu0():
                            self.build_model_single_gpu(idx, inputs, labels)
                            if not self.cross_validation:
                                d_grads = d_opt.compute_gradients(
                                        self.d_losses[-1], var_list=self.d_vars)
                                g_grads = g_opt.compute_gradients(
                                        self.g_losses[-1], var_list=self.g_vars)
                                all_d_grads.append(d_grads)
                                all_g_grads.append(g_grads)
                                tf.get_variable_scope().reuse_variables()

        if not self.cross_validation:
            avg_d_grads = average_gradients(all_d_grads)
            avg_g_grads = average_gradients(all_g_grads)
            self.d_opt = d_opt.apply_gradients(avg_d_grads)
            self.g_opt = g_opt.apply_gradients(avg_g_grads)

    def build_model_single_gpu(self, gpu_idx, inputs, labels):
        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_clean = []
            self.gtruth_noise = []

        self.gtruth_noise.append(inputs)
        self.gtruth_clean.append(labels)

        # add channels dimension to manipulate in D and G
        inputs = tf.expand_dims(inputs, -1)
        labels = tf.expand_dims(labels, -1)
        # by default leaky relu is used
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            ref_Gs = self.generator(inputs, is_ref=True,
                                    units=self.output_dim, spk=None,
                                    do_prelu=do_prelu)
            print('num of G returned: ', len(ref_Gs))
            self.reference_G = ref_Gs[0]
            self.ref_z = ref_Gs[1]
            if do_prelu:
                self.ref_alpha = ref_Gs[2:]
                self.alpha_sum = []
                # for m, ref_alpha in enumerate(self.ref_alpha):
                #     # add a summary per alpha
                #     self.alpha_sum.append(histogram_summary('alpha_{}'.format(m),
                #                                              ref_alpha))
            # make a dummy copy of discriminator to have variables and then
            # be able to set up the variable reuse for all other devices
            # merge along channels and this would be a real batch
            inputs_sq = tf.squeeze(inputs, -1)
            labels_sq = tf.squeeze(labels, -1)
            dummy_joint = tf.concat([inputs_sq, labels_sq], -1)
            dummy = discriminator(self, dummy_joint, reuse=False)

        G, z  = self.generator(inputs, is_ref=False,
                               units=self.output_dim, spk=None,
                               do_prelu=do_prelu)
        self.Gs.append(G)
        self.zs.append(z)

        # add new dimension to merge with other pairs
        inputs_sq = tf.squeeze(inputs, -1)
        labels_sq = tf.squeeze(labels, -1)
        G_sq = tf.squeeze(G, -1)
        D_rl_joint = tf.concat([inputs_sq, labels_sq], -1)
        D_fk_joint = tf.concat([inputs_sq, G_sq], -1)
        D_rl_joint = tf.expand_dims(D_rl_joint, -1)
        D_fk_joint = tf.expand_dims(D_fk_joint, -1)
        # build rl discriminator
        d_rl_logits = discriminator(self, D_rl_joint, reuse=True)
        # build fk G discriminator
        d_fk_logits = discriminator(self, D_fk_joint, reuse=True)

        # make disc variables summaries
        self.d_rl_sum = histogram_summary("d_real", d_rl_logits)
        self.d_fk_sum = histogram_summary("d_fake", d_fk_logits)
        self.real_clean_sum = histogram_summary('real_clean', inputs)
        self.real_noise_sum = histogram_summary('real_noise', labels)
        self.gen_sum = histogram_summary('G_clean', G)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            self.g_adv_losses = []
            self.d_rl_losses = []
            self.d_fk_losses = []
            self.d_losses = []

        d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
        d_fk_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))
        g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))

        d_loss = d_rl_loss + d_fk_loss

        # Add the L1 loss to G
        g_l1_loss = self.l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(G,
                                                                labels)))
        g_loss = g_adv_loss + g_l1_loss

        self.g_l1_losses.append(g_l1_loss)
        self.g_adv_losses.append(g_adv_loss)
        self.g_losses.append(g_loss)
        self.d_rl_losses.append(d_rl_loss)
        self.d_fk_losses.append(d_fk_loss)
        self.d_losses.append(d_loss)

        self.d_rl_loss_sum = scalar_summary("d_rl_loss", d_rl_loss)
        self.d_fk_loss_sum = scalar_summary("d_fk_loss", d_fk_loss)
        self.g_loss_sum = scalar_summary("g_loss", g_loss)
        self.g_loss_l1_sum = scalar_summary("g_l1_loss", g_l1_loss)
        self.g_loss_adv_sum = scalar_summary("g_adv_loss", g_adv_loss)
        self.d_loss_sum = scalar_summary("d_loss", d_loss)
        g_sums = [self.d_fk_sum,
                  self.d_fk_loss_sum,
                  self.g_loss_sum,
                  self.g_loss_l1_sum,
                  self.g_loss_adv_sum,
                  self.gen_sum]
        # if we have prelus, add them to summary
        if hasattr(self, 'alpha_sum'):
            g_sums += self.alpha_sum
        self.g_sum = tf.summary.merge(g_sums)
        self.d_sum = tf.summary.merge([self.d_loss_sum,
                                       self.d_rl_sum,
                                       self.d_rl_loss_sum,
                                       self.real_clean_sum,
                                       self.real_noise_sum,
                                       self.disc_noise_std_sum])


        if gpu_idx == 0:
            self.get_vars()

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
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars]
        else:
            print('Not clipping D weights')

    def vbn(self, tensor, name):
        if self.disable_vbn:
            class Dummy(object):
                # Do nothing here, no bnorm
                def __init__(self, tensor, ignored):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)
