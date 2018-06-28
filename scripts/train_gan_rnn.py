#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Train GAN model using new input pipeline API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from io_funcs.kaldi_io import ArkWriter
from io_funcs.tfrecords_dataset import get_batch, get_padded_batch
from models.gan_rnn import GAN_RNN
from utils.misc import *
from utils.ops import *


def train_one_iteration(sess, model, tr_num_batch, iteration):
  """ Runs the model one iteration on given data. """
  counter = 0
  d_counter = 0
  g_counter = 0
  duration = 0.0
  tr_g_loss = 0.0
  tr_d_loss = 0.0
  tr_g_adv_loss = 0.0
  tr_g_mse_loss = 0.0
  tr_g_l2_loss = 0.0
  tr_d_rl_loss = 0.0
  tr_d_fk_loss = 0.0
  soft_range = 0.2
  soft_range = soft_range * (0.98 ** iteration)
  # We run D and G alternately, so we need divide (disc_updates+gen_updates)
  sess.run(tf.assign(model.d_real, 1.0))
  sess.run(tf.assign(model.d_fake, 0.0))
  steps = model.disc_updates + model.gen_updates
  for batch in range(int(tr_num_batch/steps/FLAGS.num_gpu)):
    # D iterations
    for d_step in range(model.disc_updates):
      # d_real = np.random.uniform(1 - soft_range, 1 + soft_range)
      # d_fake = np.random.uniform(0 - soft_range, 0 + soft_range)
      # sess.run(tf.assign(model.d_real, d_real))
      # sess.run(tf.assign(model.d_fake, d_fake))
      _d_opt, d_rl_losses, \
      d_fk_losses, d_losses = sess.run([model.d_opt,
                                        model.d_rl_losses,
                                        model.d_fk_losses,
                                        model.d_losses])
      if model.d_clip_weights:
        sess.run(model.d_clip)
      d_rl_loss = np.mean(d_rl_losses)
      d_fk_loss = np.mean(d_fk_losses)
      d_loss = np.mean(d_losses)
      d_counter += 1
      tr_d_rl_loss += d_rl_loss
      tr_d_fk_loss += d_fk_loss
      tr_d_loss += d_loss
    # now G iterations
    for g_step in range(model.gen_updates):
      _g_opt, g_adv_losses, \
      g_mse_losses, g_l2_losses, \
      g_losses = sess.run([model.g_opt,
                           model.g_adv_losses,
                           model.g_mse_losses,
                           model.g_l2_losses,
                           model.g_losses])
      if model.g_disturb_weights:
        sess.run(model.g_disturb)
      g_adv_loss = np.mean(g_adv_losses)
      g_mse_loss = np.mean(g_mse_losses)
      g_l2_loss = np.mean(g_l2_losses)
      g_loss = np.mean(g_losses)
      g_counter += 1
      tr_g_adv_loss += g_adv_loss
      tr_g_mse_loss += g_mse_loss
      tr_g_l2_loss += g_l2_loss
      tr_g_loss += g_loss

    counter = (d_counter + g_counter) * FLAGS.num_gpu

  # Save summary
  _summaries = sess.run(model.summaries)
  model.writer.add_summary(_summaries, iteration*tr_num_batch)

  tr_d_rl_loss = tr_d_rl_loss / d_counter
  tr_d_fk_loss = tr_d_fk_loss / d_counter
  tr_d_loss = tr_d_loss / d_counter
  tr_g_adv_loss = tr_g_adv_loss / g_counter
  tr_g_mse_loss = tr_g_mse_loss / g_counter
  tr_g_l2_loss = tr_g_l2_loss / g_counter
  tr_g_loss = tr_g_loss / g_counter

  return tr_d_rl_loss, tr_d_fk_loss, tr_d_loss, \
         tr_g_adv_loss, tr_g_mse_loss, tr_g_l2_loss, tr_g_loss


def eval_one_iteration(sess, model, cv_num_batch, iteration):
  """ Cross validate the model on given data. """
  d_counter = 0
  g_counter = 0
  cv_g_adv_loss = 0.0
  cv_g_mse_loss = 0.0
  cv_g_l2_loss = 0.0
  cv_g_loss = 0.0
  cv_d_rl_loss = 0.0
  cv_d_fk_loss = 0.0
  cv_d_loss = 0.0
  # We run D and G alternately, so we need divide (disc_updates+gen_updates)
  sess.run(tf.assign(model.d_real, 1.0))
  sess.run(tf.assign(model.d_fake, 0.0))
  steps = model.disc_updates + model.gen_updates
  for batch in range(int(cv_num_batch/steps/FLAGS.num_gpu)):
    # D iterations
    for d_step in range(model.disc_updates):
      d_rl_losses, \
      d_fk_losses, d_losses = sess.run([model.d_rl_losses,
                                        model.d_fk_losses,
                                        model.d_losses])
      d_rl_loss = np.mean(d_rl_losses)
      d_fk_loss = np.mean(d_fk_losses)
      d_loss = np.mean(d_losses)
      d_counter += 1
      cv_d_rl_loss += d_rl_loss
      cv_d_fk_loss += d_fk_loss
      cv_d_loss += d_loss

    # now G iterations
    for g_step in range(model.gen_updates):
      g_adv_losses, g_mse_losses, \
      g_l2_losses, g_losses = sess.run([model.g_adv_losses,
                                        model.g_mse_losses,
                                        model.g_l2_losses,
                                        model.g_losses])
      g_adv_loss = np.mean(g_adv_losses)
      g_mse_loss = np.mean(g_mse_losses)
      g_l2_loss = np.mean(g_l2_losses)
      g_loss = np.mean(g_losses)
      g_counter += 1
      cv_g_adv_loss += g_adv_loss
      cv_g_mse_loss += g_mse_loss
      cv_g_l2_loss += g_l2_loss
      cv_g_loss += g_loss

  _summaries = sess.run(model.summaries)
  model.writer.add_summary(_summaries, iteration*cv_num_batch)

  cv_d_rl_loss = cv_d_rl_loss / d_counter
  cv_d_fk_loss = cv_d_fk_loss / d_counter
  cv_d_loss = cv_d_loss / d_counter
  cv_g_adv_loss = cv_g_adv_loss / g_counter
  cv_g_mse_loss = cv_g_mse_loss / g_counter
  cv_g_l2_loss = cv_g_l2_loss / g_counter
  cv_g_loss = cv_g_loss / g_counter

  return cv_d_rl_loss, cv_d_fk_loss, cv_d_loss, \
         cv_g_adv_loss, cv_g_mse_loss, cv_g_l2_loss, cv_g_loss


def decode():
  """Decoding the inputs using current model."""
  tf.logging.info("Get TEST sets number.")
  num_batch = get_num_batch(FLAGS.test_list_file, infer=True)
  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        data_list = read_list(FLAGS.test_list_file)
        test_utt_id, test_inputs, \
        test_lengths = get_batch(data_list,
                                 batch_size=1,
                                 input_size=FLAGS.input_dim,
                                 output_size=FLAGS.output_dim,
                                 left_context=FLAGS.left_context,
                                 right_context=FLAGS.right_context,
                                 num_threads=FLAGS.num_threads,
                                 num_epochs=1,
                                 infer=True)

    devices = []
    for i in xrange(FLAGS.num_gpu):
      device_name = ("/gpu:%d" % i)
      print('Using device: ', device_name)
      devices.append(device_name)

    # Prevent exhausting all the gpu memories.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # execute the session
    with tf.Session(config=config) as sess:
      # Create two models with tr_inputs and cv_inputs individually.
      with tf.name_scope('model'):
        model = GAN_RNN(sess, FLAGS, devices, test_inputs, labels=None,
                        lengths=test_lengths, cross_validation=True)

      show_all_variables()

      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
      print("Initializing variables ...")
      sess.run(init)

      if model.load(model.save_dir, moving_average=False):
        print("[*] Load SUCCESS")
      else:
        print("[!] Load failed. Checkpoint not found. Exit now.")
        sys.exit(1)

      cmvn_filename = os.path.join(FLAGS.data_dir, "train_cmvn.npz")
      if os.path.isfile(cmvn_filename):
        cmvn = np.load(cmvn_filename)
      else:
        tf.logging.fatal("%s not exist, exit now." % cmvn_filename)
        sys.exit(1)

      out_dir_name = os.path.join(FLAGS.save_dir, 'test')
      if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

      write_scp_path = os.path.join(out_dir_name, 'feats.scp')
      write_ark_path = os.path.join(out_dir_name, 'feats.ark')
      writer = ArkWriter(write_scp_path)

      start = datetime.datetime.now()
      outputs = model.generator(test_inputs, None, test_lengths, reuse=True)
      outputs = tf.reshape(outputs, [-1, model.output_dim])
      for batch in range(num_batch):
        try:
          utt_id, activations = sess.run([test_utt_id, outputs])
          sequence = activations * cmvn['stddev_labels'] + \
                  cmvn['mean_labels']
          save_result = np.vstack(sequence)
          writer.write_next_utt(write_ark_path, utt_id[0], save_result)
          print("[{}/{}] Write inferred {} to {}".format(
              batch+1, num_batch, utt_id[0], write_ark_path))
        except tf.errors.OutOfRangeError:
          tf.logging.error("Out of range error occured when decoding.")
          break

      sess.close()
      end = datetime.datetime.now()
      duration = (end - start).total_seconds()
      print("Decoding time is {}s".format(duration))
      sys.stdout.flush()

    tf.logging.info("Decoding Done.")


def main(_):
  if not FLAGS.decode:
    filename = ("batch_num_frame_%s.txt" % FLAGS.batch_size)
    batch_file = os.path.join(FLAGS.data_dir, filename)
    if os.path.isfile(batch_file):
      with open(batch_file, 'r') as fr:
        line = fr.readline()
        line = line.strip().split()
        cv_num_batch = int(line[0])
        tr_num_batch = int(line[1])
        print('LOG: %s exist, cross validation batches is %d, trian '
              'batches is %d.' % (filename, cv_num_batch, tr_num_batch))
    else:
      print("Get CV set batch numbers.")
      cv_num_batch = get_num_batch(FLAGS.cv_list_file, infer=False)
      print("Get Train set batch numbers.")
      tr_num_batch = get_num_batch(FLAGS.tr_list_file, infer=False)
      with open(batch_file, 'w') as fw:
        fw.write("%d %d" % (cv_num_batch, tr_num_batch))
    # train_batch_per_iter = 500 * (32 / FLAGS.batch_size)
    # valdi_batch_per_iter = 70 * (32 / FLAGS.batch_size)
    train_batch_per_iter = tr_num_batch
    valdi_batch_per_iter = cv_num_batch
    if train_batch_per_iter > tr_num_batch: train_batch_per_iter = tr_num_batch
    if valdi_batch_per_iter > cv_num_batch: valdi_batch_per_iter = cv_num_batch
    min_iters = int(FLAGS.min_epoches * tr_num_batch / train_batch_per_iter)
    max_iters = int(FLAGS.max_epoches * tr_num_batch / train_batch_per_iter)
    print("\nLOG: #train_batch = {}, #valid_batch = {}\n"
          "LOG: #batch_per_train_iter = {}, #batch_per_valid_iter = {}\n"
          "LOG: #min_epoches = {}, #max_epoches = {}\n"
          "LOG: #min_iters = {}, #max_iters = {}, #itres_per_epoch = {:.2f}"
          "\n".format(
              tr_num_batch, cv_num_batch,
              train_batch_per_iter, valdi_batch_per_iter,
              FLAGS.min_epoches, FLAGS.max_epoches,
              min_iters, max_iters, max_iters/FLAGS.max_epoches))
    train(valdi_batch_per_iter, train_batch_per_iter, min_iters, max_iters)
  else:
    decode()


def get_num_batch(file_list, infer=False):
  """ Get number of bacthes. """
  data_list = read_list(file_list)
  counter = 0

  with tf.Graph().as_default():
    if not infer:
      _, inputs, labels, \
      lengths = get_padded_batch(data_list,
                                 FLAGS.batch_size,
                                 FLAGS.input_dim,
                                 FLAGS.output_dim, 0, 0,
                                 FLAGS.num_threads*2, 1)
    else:
      utt_id, inputs, \
      lengths = get_batch(data_list,
                          FLAGS.batch_size,
                          FLAGS.input_dim,
                          FLAGS.output_dim, 0, 0,
                          FLAGS.num_threads*2, 1,
                          infer=infer)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    start = datetime.datetime.now()
    while True:
      try:
        sess.run([inputs])
        counter += 1
      except tf.errors.OutOfRangeError:
        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print('Number of batches is %d. Reading time is %.0fs.' % (
            counter, duration))
        break

    sess.close()
  return counter


def train(valdi_batch_per_iter, train_batch_per_iter, min_iters, max_iters):
  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        tr_data_list = read_list(FLAGS.tr_list_file)
        _, tr_inputs, tr_labels, \
        tr_lengths = get_padded_batch(tr_data_list,
                                      FLAGS.batch_size,
                                      FLAGS.input_dim,
                                      FLAGS.output_dim,
                                      FLAGS.left_context,
                                      FLAGS.right_context,
                                      FLAGS.num_threads,
                                      FLAGS.max_epoches)
        tr_inputs.set_shape([None, None,
            FLAGS.input_dim * (FLAGS.left_context + 1 + FLAGS.right_context)])

        cv_data_list = read_list(FLAGS.cv_list_file)
        _, cv_inputs, cv_labels, \
        cv_lengths = get_padded_batch(cv_data_list,
                                      FLAGS.batch_size,
                                      FLAGS.input_dim,
                                      FLAGS.output_dim,
                                      FLAGS.left_context,
                                      FLAGS.right_context,
                                      FLAGS.num_threads,
                                      None)
        cv_inputs.set_shape([None, None,
            FLAGS.input_dim * (FLAGS.left_context + 1 + FLAGS.right_context)])

    devices = []
    for i in xrange(FLAGS.num_gpu):
      device_name = ("/gpu:%d" % i)
      print('Using device: ', device_name)
      devices.append(device_name)
    # Prevent exhausting all the gpu memories.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # execute the session
    with tf.Session(config=config) as sess:
      # Create two models with tr_inputs and cv_inputs individually.
      with tf.name_scope('model'):
        print("=======================================================")
        print("|                Build Train model                    |")
        print("=======================================================")
        tr_model = GAN_RNN(sess, FLAGS, devices, tr_inputs, tr_labels,
                           tr_lengths, cross_validation=False)
        # tr_model and val_model should share variables
        print("=======================================================")
        print("|           Build Cross-Validation model              |")
        print("=======================================================")
        tf.get_variable_scope().reuse_variables()
        cv_model = GAN_RNN(sess, FLAGS, devices, cv_inputs, cv_labels,
                           cv_lengths, cross_validation=True)

      show_all_variables()

      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
      print("Initializing variables ...")
      sess.run(init)

      if tr_model.load(tr_model.save_dir, moving_average=False):
        print("[*] Load SUCCESS")
      else:
        print("[!] Begin a new model.")
      sys.stdout.flush()

      # Early stop counter
      g_loss_prev = 10000.0
      g_rel_impr = 1.0
      check_interval = 1
      windows_g_loss = []

      g_learning_rate = FLAGS.num_gpu * FLAGS.g_learning_rate
      d_learning_rate = FLAGS.num_gpu * FLAGS.d_learning_rate
      sess.run(tf.assign(tr_model.g_learning_rate, g_learning_rate))
      sess.run(tf.assign(tr_model.d_learning_rate, d_learning_rate))

      for iteration in range(max_iters):
        try:
          start = datetime.datetime.now()
          tr_d_rl_loss, tr_d_fk_loss, \
          tr_d_loss, tr_g_adv_loss, \
          tr_g_mse_loss, tr_g_l2_loss, \
          tr_g_loss = train_one_iteration(sess, tr_model,
                                          train_batch_per_iter, iteration+1)
          cv_d_rl_loss, cv_d_fk_loss, \
          cv_d_loss, cv_g_adv_loss, \
          cv_g_mse_loss, cv_g_l2_loss, \
          cv_g_loss = eval_one_iteration(sess, cv_model,
                                         valdi_batch_per_iter, iteration+1)
          d_learning_rate, \
          g_learning_rate = sess.run([tr_model.d_learning_rate,
                                      tr_model.g_learning_rate])

          end = datetime.datetime.now()
          print("{}/{} (INFO): d_learning_rate = {:.5e}, "
                "g_learning_rate = {:.5e}, time = {:.3f} h\n"
                "{}/{} (TRAIN AVG.LOSS): "
                "d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, "
                "d_loss = {:.5f}, g_adv_loss = {:.5f}, "
                "g_mse_loss = {:.5f}, g_l2_loss = {:.3e}, "
                "g_loss = {:.5f}\n"
                "{}/{} (CROSS AVG.LOSS): "
                "d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, "
                "d_loss = {:.5f}, g_adv_loss = {:.5f}, "
                "g_mse_loss = {:.5f}, g_l2_loss = {:.3e}, "
                "g_loss = {:.5f}".format(
                    iteration+1, max_iters, d_learning_rate,
                    g_learning_rate, (end-start).total_seconds()/3600.0,
                    iteration+1, max_iters,
                    tr_d_rl_loss, tr_d_fk_loss,
                    tr_d_loss, tr_g_adv_loss,
                    tr_g_mse_loss, tr_g_l2_loss,
                    tr_g_loss,
                    iteration+1, max_iters,
                    cv_d_rl_loss, cv_d_fk_loss,
                    cv_d_loss, cv_g_adv_loss,
                    cv_g_mse_loss, cv_g_l2_loss,
                    cv_g_loss))
          sys.stdout.flush()

          # Start decay learning rate
          g_learning_rate = exponential_decay(iteration+1, FLAGS.num_gpu,
                                              min_iters, FLAGS.g_learning_rate)
          d_learning_rate = exponential_decay(iteration+1, FLAGS.num_gpu,
                                              min_iters, FLAGS.d_learning_rate)
          disc_noise_std = exponential_decay(iteration+1, FLAGS.num_gpu,
              min_iters, FLAGS.init_disc_noise_std, multiply_jobs=False)
          sess.run(tf.assign(tr_model.g_learning_rate, g_learning_rate))
          sess.run(tf.assign(tr_model.d_learning_rate, d_learning_rate))
          sess.run(tf.assign(tr_model.disc_noise_std, disc_noise_std))

          windows_g_loss.append(cv_g_loss)

          # Accept or reject new parameters.
          if (iteration + 1) % check_interval == 0:
            g_loss_new = np.mean(windows_g_loss)
            g_rel_impr = (g_loss_prev - g_loss_new) / g_loss_prev
            if g_rel_impr > 0.0:
              tr_model.save(tr_model.save_dir, iteration+1)
              print("Iteration {}: Nnet Accepted. "
                    "Save model SUCCESS. g_loss_prev = {:.5f}, "
                    "g_loss_new = {:.5f}".format(iteration+1,
                        g_loss_prev, g_loss_new))
              g_loss_prev = g_loss_new
            else:
              print("Iteration {}: Nnet Rejected. "
                    "g_loss_prev = {:.5f}, "
                    "g_loss_new = {:.5f}".format(iteration+1,
                        g_loss_prev, g_loss_new))
              # tr_model.load(tr_model.save_dir, moving_average=False)
            windows_g_loss = []

          # Stopping criterion.
          if iteration + 1 > min_iters and \
              (iteration + 1) % check_interval == 0:
            if g_rel_impr < FLAGS.end_improve:
              print("Iteration %d: Finished, too small relative "
                    "G improvement %g" % (iteration+1, g_rel_impr))
              break
          sys.stdout.flush()

        except tf.errors.OutOfRangeError:
          tf.logging.error("Out of range occured when training.")
          break

      # Whether save the last model
      if windows_g_loss:
        g_loss_new = np.mean(windows_g_loss)
        g_rel_impr = (g_loss_prev - g_loss_new) / g_loss_prev
        if g_rel_impr > 0.0:
          tr_model.save(tr_model.save_dir, iteration+1)
          print("Iteration {}: Nnet Accepted. "
                "Save model SUCCESS. g_loss_prev = {:.5f}, "
                "g_loss_new = {:.5f}".format(iteration+1,
                      g_loss_prev, g_loss_new))
          g_loss_prev = g_loss_new
          sys.stdout.flush()
        windows_g_loss = []

    sess.close()
    tf.logging.info("Training Done.")


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--decode",
      default=False,
      action="store_true",
      help="Flag indicating decoding or training."
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default=None,
      help="Data directory."
  )
  parser.add_argument(
      "--tr_list_file",
      type=str,
      default=None,
      help="TFRecords train set list file."
  )
  parser.add_argument(
      "--cv_list_file",
      type=str,
      default=None,
      help="TFRecords validation set list file."
  )
  parser.add_argument(
      "--test_list_file",
      type=str,
      default=None,
      help="TFRecords test set list file."
  )
  parser.add_argument(
      "--input_dim",
      type=int,
      default=257,
      help="The dimension of input."
  )
  parser.add_argument(
      "--output_dim",
      type=int,
      default=40,
      help="The dimension of output."
  )
  parser.add_argument(
      "--left_context",
      type=int,
      default=5,
      help="The number of left context to be added to inputs."
  )
  parser.add_argument(
      "--right_context",
      type=int,
      default=5,
      help="The number of right context to be added to inputs."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Mini-batch size."
  )
  parser.add_argument(
      "--g_learning_rate",
      type=float,
      default=0.0003,
      help="Initial G learning rate."
  )
  parser.add_argument(
      "--d_learning_rate",
      type=float,
      default=0.001,
      help="Initial D learning rate."
  )
  parser.add_argument(
      "--min_epoches",
      type=int,
      default=25,
      help="Min number of epoches to run trainer without decay."
  )
  parser.add_argument(
      "--max_epoches",
      type=int,
      default=30,
      help="Max number of epoches to run trainer totally."
  )
  parser.add_argument(
      "--end_improve",
      type=float,
      default=0.001,
      help="Stop when relative loss is lower than end_improve."
  )
  parser.add_argument(
      "--num_threads",
      type=int,
      default=24,
      help='The num of threads to read tfrecords files.'
  )
  parser.add_argument(
      "--save_dir",
      type=str,
      default="exp/gan_rnn",
      help="Directory to put the train result."
  )
  parser.add_argument(
      "--init_mse_weight",
      type=float,
      default=1.0,
      help="Init MSE loss lambda."
  )
  parser.add_argument(
      "--g_type",
      type=str,
      default="lstm",
      help="Type of G to use: lstm"
  )
  parser.add_argument(
      "--disc_updates",
      type=int,
      default=1,
      help="Number of D step in a training iteration."
  )
  parser.add_argument(
      "--gen_updates",
      type=int,
      default=2,
      help="Number of G step in a training iteration."
  )
  parser.add_argument(
      '--batch_norm',
      type=str2bool,
      nargs='?',
      default='false',
      help="Whether use batch normalization."
  )
  parser.add_argument(
      "--keep_prob",
      type=float,
      default=1.0,
      help="The probability that each element is kept for dropout."
  )
  parser.add_argument(
      "--init_disc_noise_std",
      type=float,
      default=0.0,
      help="Noise std for discriminator."
  )
  parser.add_argument(
      '--l2_scale',
      type=float,
      default=0.00001,
      help="Scale used for L2 regularizer."
  )
  parser.add_argument(
      "--num_gpu",
      type=int,
      default=1,
      help="Number of GPU to use."
  )

  print('*** Parsed arguments ***')
  FLAGS, unparsed = parser.parse_known_args()
  pp.pprint(FLAGS.__dict__)
  sys.stdout.flush()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
