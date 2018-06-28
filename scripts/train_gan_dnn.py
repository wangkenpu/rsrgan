#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

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
from io_funcs.tfrecords_io import get_batch
from models.gan import GAN
from utils.misc import *
from utils.ops import *


def train_one_epoch(sess, coord, model, tr_num_batch, epoch):
    """ Runs the model one epoch on given data. """
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
    d_counter_report = 0
    g_counter_report = 0
    tr_g_loss_report = 0.0
    tr_d_loss_report = 0.0
    tr_g_adv_loss_report = 0.0
    tr_g_mse_loss_report = 0.0
    tr_g_l2_loss_report = 0.0
    tr_d_rl_loss_report = 0.0
    tr_d_fk_loss_report = 0.0

    start = datetime.datetime.now()
    # We run D and G alternately, so we need divide (disc_updates+gen_updates)
    steps = model.disc_updates + model.gen_updates
    for batch in xrange(int(tr_num_batch/steps/FLAGS.num_gpu)):
        if coord.should_stop():
            break
        # D iterations
        for d_step in range(model.disc_updates):
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
            d_counter_report += 1
            tr_d_rl_loss_report += d_rl_loss
            tr_d_fk_loss_report += d_fk_loss
            tr_d_loss_report += d_loss
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
            g_counter_report += 1
            tr_g_adv_loss_report += g_adv_loss
            tr_g_mse_loss_report += g_mse_loss
            tr_g_l2_loss_report += g_l2_loss
            tr_g_loss_report += g_loss

        counter = (d_counter + g_counter) * FLAGS.num_gpu

        if (counter/FLAGS.num_gpu) % 1000 == 0:
            end = datetime.datetime.now()
            duration = (end - start).total_seconds()
            print("Epoch {} (BATCH {}): "
                  "d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, "
                  "d_loss = {:.5f}, g_adv_loss = {:.5f}, "
                  "g_mse_loss = {:.5f}, g_l2_loss = {:.5f}, "
                  "g_loss = {:.5f}, time = {:.3} min".format(
                       epoch, counter,
                       tr_d_rl_loss_report/d_counter_report,
                       tr_d_fk_loss_report/d_counter_report,
                       tr_d_loss_report/d_counter_report,
                       tr_g_adv_loss_report/g_counter_report,
                       tr_g_mse_loss_report/g_counter_report,
                       tr_g_l2_loss_report/g_counter_report,
                       tr_g_loss_report/g_counter_report,
                       duration/60.0))
            start = datetime.datetime.now()
            sys.stdout.flush()
            d_counter_report = 0
            g_counter_report = 0
            tr_g_loss_report = 0.0
            tr_d_loss_report = 0.0
            tr_g_adv_loss_report = 0.0
            tr_g_mse_loss_report = 0.0
            tr_g_l2_loss_report = 0.0
            tr_d_rl_loss_report = 0.0
            tr_d_fk_loss_report = 0.0

    # Save summary
    _summaries = sess.run(model.summaries)
    model.writer.add_summary(_summaries, epoch*tr_num_batch)

    tr_d_rl_loss = tr_d_rl_loss / d_counter
    tr_d_fk_loss = tr_d_fk_loss / d_counter
    tr_d_loss = tr_d_loss / d_counter
    tr_g_adv_loss = tr_g_adv_loss / g_counter
    tr_g_mse_loss = tr_g_mse_loss / g_counter
    tr_g_l2_loss = tr_g_l2_loss / g_counter
    tr_g_loss = tr_g_loss / g_counter

    return tr_d_rl_loss, tr_d_fk_loss, tr_d_loss, \
           tr_g_adv_loss, tr_g_mse_loss, tr_g_l2_loss, tr_g_loss


def eval_one_epoch(sess, coord, model, cv_num_batch, epoch):
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
    steps = model.disc_updates + model.gen_updates
    for batch in xrange(int(cv_num_batch/steps/FLAGS.num_gpu)):
        if coord.should_stop():
            break
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
    model.writer.add_summary(_summaries, epoch*cv_num_batch)

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
                test_utt_id, test_inputs, _ = get_batch(
                        data_list,
                        batch_size=1,
                        input_size=FLAGS.input_dim,
                        output_size=FLAGS.output_dim,
                        left=FLAGS.left_context,
                        right=FLAGS.right_context,
                        num_enqueuing_threads=FLAGS.num_threads,
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
                model = GAN(sess, FLAGS, devices, test_inputs, labels=None,
                            cross_validation=True)

            show_all_variables()

            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            print("Initializing variables ...")
            sess.run(init)

            if model.load(model.save_dir, moving_average=True):
                print("[*] Load SUCCESS")
            else:
                print("[!] Load failed. Checkpoint not found. Exit now.")
                sys.exit(1)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

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

            try:
                for batch in range(num_batch):
                    if coord.should_stop():
                        break
                    outputs = model.generator(test_inputs, None, reuse=True)
                    outputs = tf.reshape(outputs, [-1, model.output_dim])
                    utt_id, activations = sess.run([test_utt_id, outputs])
                    sequence = activations * cmvn['stddev_labels'] + \
                            cmvn['mean_labels']
                    save_result = np.vstack(sequence)
                    writer.write_next_utt(write_ark_path, utt_id[0], save_result)
                    tf.logging.info("Write inferred %s to %s" %
                            (utt_id[0], write_ark_path))
            except Exception, e:
                # Report exceptions to the coordinator.
                coord.request_stop(e)
            finally:
                # Terminate as usual.  It is innocuous to request stop twice.
                coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)

        tf.logging.info("Decoding Done.")

def main(_):
    if not FLAGS.decode:
        batch_file = os.path.join(FLAGS.data_dir, 'batch_num.txt')
        if os.path.isfile(batch_file):
            with open(batch_file, 'r') as fr:
                line = fr.readline()
                line = line.strip().split()
                cv_num_batch = int(line[0])
                tr_num_batch = int(line[1])
                print('[*] batch_num.txt exist, and cv batches is %d, '
                      'tr batches is %d.' % (cv_num_batch, tr_num_batch))
        else:
            print("Get CV set batch numbers.")
            cv_num_batch = get_num_batch(FLAGS.cv_list_file, infer=False)
            print("Get Train set batch numbers.")
            tr_num_batch = get_num_batch(FLAGS.tr_list_file, infer=False)
            with open(batch_file, 'w') as fw:
                fw.write("%d %d" % (cv_num_batch, tr_num_batch))

        train(cv_num_batch, tr_num_batch)
    else:
        decode()


def get_num_batch(file_list, infer=False):
    """ Get number of bacthes. """
    data_list = read_list(file_list)
    counter = 0

    with tf.Graph().as_default():
        if not infer:
            inputs, labels = get_batch(data_list,
                                       FLAGS.batch_size,
                                       FLAGS.input_dim,
                                       FLAGS.output_dim, 0, 0,
                                       FLAGS.num_threads*2, 1,
                                       infer=infer)
        else:
            utt_id, inputs, _ = get_batch(data_list,
                                          FLAGS.batch_size,
                                          FLAGS.input_dim,
                                          FLAGS.output_dim, 0, 0,
                                          FLAGS.num_threads*2, 1,
                                          infer=infer)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            start = datetime.datetime.now()
            while not coord.should_stop():
                sess.run([inputs])
                counter += 1
        except tf.errors.OutOfRangeError:
            end = datetime.datetime.now()
            duration = (end - start).total_seconds()
            print('Number of batches is %d. Reading time is %.0fs.' % (
                counter, duration))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

    return counter


def train(cv_num_batch, tr_num_batch):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                tr_data_list = read_list(FLAGS.tr_list_file)
                tr_inputs, tr_labels = get_batch(tr_data_list,
                                                 FLAGS.batch_size,
                                                 FLAGS.input_dim,
                                                 FLAGS.output_dim,
                                                 FLAGS.left_context,
                                                 FLAGS.right_context,
                                                 FLAGS.num_threads,
                                                 FLAGS.max_epoches)

                cv_data_list = read_list(FLAGS.cv_list_file)
                cv_inputs, cv_labels = get_batch(cv_data_list,
                                                 FLAGS.batch_size,
                                                 FLAGS.input_dim,
                                                 FLAGS.output_dim,
                                                 FLAGS.left_context,
                                                 FLAGS.right_context,
                                                 FLAGS.num_threads,
                                                 FLAGS.max_epoches)

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
                tr_model = GAN(sess, FLAGS, devices, tr_inputs, tr_labels,
                               cross_validation=False)
                # tr_model and val_model should share variables
                print("=======================================================")
                print("|           Build Cross-Validation model              |")
                print("=======================================================")
                tf.get_variable_scope().reuse_variables()
                cv_model = GAN(sess, FLAGS, devices, cv_inputs, cv_labels,
                               cross_validation=True)

            show_all_variables()

            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            print("Initializing variables ...")
            sess.run(init)

            if tr_model.load(tr_model.save_dir):
                print("[*] Load SUCCESS")
            else:
                print("[!] Begin a new model.")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                cv_d_rl_loss, cv_d_fk_loss, \
                cv_d_loss, cv_g_adv_loss, \
                cv_g_mse_loss, cv_g_l2_loss, \
                cv_g_loss = eval_one_epoch(sess, coord, cv_model, cv_num_batch, 0)
                print("CROSSVAL.LOSS PRERUN: "
                      "d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, "
                      "d_loss = {:.5f}, g_adv_loss = {:.5f}, "
                      "g_mse_loss = {:.5f}, g_l2_loss = {:.5f}, "
                      "g_loss = {:.5f}".format(cv_d_rl_loss,
                          cv_d_fk_loss, cv_d_loss, cv_g_adv_loss,
                          cv_g_mse_loss, cv_g_l2_loss, cv_g_loss))
                sys.stdout.flush()

                g_loss_prev = cv_g_loss
                decay_steps = 1

                for epoch in range(FLAGS.max_epoches):
                    start = datetime.datetime.now()
                    tr_d_rl_loss, tr_d_fk_loss, \
                    tr_d_loss, tr_g_adv_loss, \
                    tr_g_mse_loss, tr_g_l2_loss, \
                    tr_g_loss = train_one_epoch(sess, coord,
                                                tr_model, tr_num_batch, epoch+1)
                    cv_d_rl_loss, cv_d_fk_loss, \
                    cv_d_loss, cv_g_adv_loss, \
                    cv_g_mse_loss, cv_g_l2_loss, \
                    cv_g_loss = eval_one_epoch(sess, coord,
                                               cv_model, cv_num_batch, epoch+1)
                    d_lr, g_lr = sess.run([tr_model.d_learning_rate,
                                           tr_model.g_learning_rate])

                    end = datetime.datetime.now()
                    print("Epoch {} (TRAIN AVG.LOSS): "
                          "d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, "
                          "d_loss = {:.5f}, g_adv_loss = {:.5f}, "
                          "g_mse_loss = {:.5f}, g_l2_loss = {:.5f}, "
                          "g_loss = {:.5f}, "
                          "d_lr = {:.3e}, g_lr = {:.3e}\n"
                          "Epoch {} (CROSS AVG.LOSS): "
                          "d_rl_loss = {:.5f}, d_fk_loss = {:.5f}, "
                          "d_loss = {:.5f}, g_adv_loss = {:.5f}, "
                          "g_mse_loss = {:.5f}, g_l2_loss = {:.5f}, "
                          "g_loss = {:.5f}, "
                          "time = {:.2f} h".format(epoch+1,
                              tr_d_rl_loss, tr_d_fk_loss, tr_d_loss,
                              tr_g_adv_loss, tr_g_mse_loss, tr_g_l2_loss,
                              tr_g_loss, d_lr, g_lr, epoch+1,
                              cv_d_rl_loss, cv_d_fk_loss, cv_d_loss,
                              cv_g_adv_loss, cv_g_mse_loss, cv_g_l2_loss,
                              cv_g_loss, (end-start).seconds/3600.0))
                    sys.stdout.flush()

                    g_loss_new = cv_g_loss
                    # Accept or reject new parameters
                    if g_loss_new < g_loss_prev:
                        tr_model.save(tr_model.save_dir, epoch+1)
                        print("Epoch {}: Nnet Accepted. "
                              "Save model SUCCESS.".format(epoch+1))
                        # Relative loss between previous and current val_loss
                        g_rel_impr = (g_loss_prev - g_loss_new) / g_loss_prev
                        g_loss_prev = g_loss_new
                    else:
                        print("Epoch {}: Nnet Rejected.".format(epoch+1))
                        if tr_model.load(tr_model.save_dir):
                            print("[*] Load previous model SUCCESS.")
                            sys.stdout.flush()
                        else:
                            print("[!] Load failed. No checkpoint from {} to "
                                  "restore previous model. Exit now.".format(
                                      tr_model.save_dir))
                            sys.stdout.flush()
                            sys.exit(1)
                        # Relative loss between previous and current val_loss
                        g_rel_impr = (g_loss_prev - g_loss_new) / g_loss_prev

                    # Start decay when improvement is low (Exponential decay)
                    if g_rel_impr < FLAGS.start_decay_impr and \
                            epoch+1 >= FLAGS.keep_lr:
                        g_learning_rate = \
                                FLAGS.g_learning_rate * \
                                FLAGS.decay_factor ** (decay_steps)
                        d_learning_rate = \
                                FLAGS.d_learning_rate * \
                                FLAGS.decay_factor ** (decay_steps)
                        disc_noise_std = \
                                FLAGS.init_disc_noise_std * \
                                FLAGS.decay_factor ** (decay_steps)
                        sess.run(tf.assign(
                            tr_model.g_learning_rate, g_learning_rate))
                        sess.run(tf.assign(
                            tr_model.d_learning_rate, d_learning_rate))
                        sess.run(tf.assign(
                            tr_model.disc_noise_std, disc_noise_std))
                        decay_steps += 1

                    # Stopping criterion
                    if g_rel_impr < FLAGS.end_decay_impr:
                        if epoch < FLAGS.min_epoches:
                            print("Epoch %d: We were supposed to finish, "
                                  "but we continue as min_epoches %d" % (
                                      epoch+1, FLAGS.min_epoches))
                            continue
                        else:
                            print("Epoch %d: Finished, too small relative "
                                  "G improvement %g" % (epoch+1, g_rel_impr))
                            break

            except Exception, e:
                # Report exceptions to the coordinator.
                coord.request_stop(e)
            finally:
                # Terminate as usual.  It is innocuous to request stop twice.
                coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)

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
        default=256,
        help="Mini-batch size."
    )
    parser.add_argument(
        "--g_learning_rate",
        type=float,
        default=0.0001,
        help="Initial G learning rate."
    )
    parser.add_argument(
        "--d_learning_rate",
        type=float,
        default=0.0001,
        help="Initial D learning rate."
    )
    parser.add_argument(
        "--min_epoches",
        type=int,
        default=15,
        help="Min number of epoches to run trainer without decay."
    )
    parser.add_argument(
	"--max_epoches",
        type=int,
        default=20,
        help="Max number of epoches to run trainer totally."
    )
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=0.8,
        help="Factor for decay learning rate."
    )
    parser.add_argument(
        "--start_decay_impr",
        type=float,
        default=0.01,
        help="Decaying when ralative loss is lower than start_decay_impr."
    )
    parser.add_argument(
        "--end_decay_impr",
        type=float,
        default=0.001,
        help="Stop when relative loss is lower than end_decay_impr."
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
        default="exp/gan",
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
        default="dnn",
        help="Type of G to use: dnn or cnn."
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
        default=1,
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
        "--init_disc_noise_std",
        type=float,
        default=0.0,
        help="Noise std for discriminator."
    )
    parser.add_argument(
        '--keep_lr',
        type=int,
        default=3,
        help="Keeping learning rate epochs."
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=1.0,
        help="The probability that each element is kept for dropout."
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
