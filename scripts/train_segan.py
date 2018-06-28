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

import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from io_funcs.tfrecords_io import get_batch
from models.segan import SEGAN
from utils.utils import pp, read_list, show_all_variables


def train_one_epoch(sess, coord, model, tr_num_batch, epoch):
    """ Runs the model one epoch on given data. """
    counter = 0
    tr_g_loss = 0.0
    tr_d_loss = 0.0
    for batch in xrange(int(tr_num_batch/2/FLAGS.num_gpu-1)):
        if coord.should_stop():
            break
        if (counter+1) % 20 == 0:
            _d_opt, _d_sum, \
            d_loss = sess.run([model.d_opt, model.d_sum,
                               model.d_losses[0]])
            if model.d_clip_weights:
                sess.run(model.d_clip)

            # now G iterationsa
            _g_opt, _g_sum, \
            g_loss = sess.run([model.g_opt, model.g_sum,
                                    model.g_losses[0]])
        else:
            _d_opt, \
            d_loss = sess.run([model.d_opt,
                               model.d_losses[0]])
            if model.d_clip_weights:
                sess.run(model.d_clip)

            # now G iterationsa
            _g_opt, \
            g_loss = sess.run([model.g_opt,
                                    model.g_losses[0]])
        tr_g_loss += g_loss
        tr_d_loss += d_loss
        counter += FLAGS.num_gpu
        if (counter) % 10 == 0:
            print(("Epoch %02d(MINIBATCH %d): TRAIN AVG.LOSS %.5f(G_loss) "
                "%.5f(D_loss)") % (epoch, counter,
                    tr_g_loss/counter*FLAGS.num_gpu,
                    tr_d_loss/counter))
            sys.stdout.flush()
    tr_g_loss = tr_g_loss / counter * FLAGS.num_gpu
    tr_d_loss = tr_d_loss / counter * FLAGS.num_gpu

    return tr_g_loss, tr_d_loss


def eval_one_epoch(sess, coord, model, cv_num_batch):
    """ Cross validate the model on given data. """
    counter = 0
    cv_g_loss = 0.0
    cv_d_loss = 0.0
    for batch in xrange(int(cv_num_batch/2/FLAGS.num_gpu-1)):
        if coord.should_stop():
            break
        d_loss = sess.run([model.d_losses[0]])
        g_loss = sess.run([model.g_losses[0]])
        counter += FLAGS.num_gpu
        cv_g_loss += g_loss[0]
        cv_d_loss += d_loss[0]
    cv_g_loss = cv_g_loss / counter * FLAGS.num_gpu
    cv_d_loss = cv_d_loss / counter * FLAGS.num_gpu
    return cv_g_loss, cv_d_loss


def main(_):
    tf.logging.info("Get CV set batch numbers.")
    cv_num_batch = get_num_batch(FLAGS.cv_list_file)
    tf.logging.info("Get Train set batch numbers.")
    tr_num_batch = get_num_batch(FLAGS.tr_list_file)
    train(cv_num_batch, tr_num_batch)


def get_num_batch(file_list):
    """ Get number of bacthes. """
    data_list = read_list(file_list)
    counter = 0

    with tf.Graph().as_default():
        inputs, labels = get_batch(data_list,
                                   FLAGS.batch_size,
                                   FLAGS.input_dim,
                                   FLAGS.output_dim, 0, 0,
                                   num_enqueuing_threads=FLAGS.num_threads)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            start = datetime.datetime.now()
            while not coord.should_stop():
                inputs_tmp, labels_tmp = sess.run([inputs, labels])
                counter += 1
        except tf.errors.OutOfRangeError:
            end = datetime.datetime.now()
            duration = (end - start).total_seconds()
            tf.logging.info('Number of batches is %d. Reading time is %.0fs.' \
                    % (counter, duration))
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
                print("                  Build Train model                    ")
                print("=======================================================")
                tr_model = SEGAN(sess, FLAGS, devices, tr_inputs, tr_labels,
                                 cross_validation=False)
                # tr_model and val_model should share variables
                print("=======================================================")
                print("            Build Cross-Validation model               ")
                print("=======================================================")
                tf.get_variable_scope().reuse_variables()
                cv_model = SEGAN(sess, FLAGS, devices, cv_inputs, cv_labels,
                                 cross_validation=True)

            show_all_variables()

            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            print("Initializing variables ...")
            sess.run(init)

            if tr_model.load(tr_model.save_dir):
                print("[*] Load SUCCESS")
            else:
                print("[!] Load failed, maybe begin a new model.")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                cv_g_loss, cv_d_loss = eval_one_epoch(sess, coord, cv_model,
                                                      cv_num_batch)
                print(("CROSSVAL PRERUN AVG.LOSS "
                    "%.4F(G_loss), %.4F(D_loss)") % (cv_g_loss, cv_d_loss))
                sys.stdout.flush()

                for epoch in xrange(FLAGS.max_epoches):
                    start = datetime.datetime.now()
                    tr_g_loss, tr_d_loss = train_one_epoch(sess, coord,
                            tr_model, tr_num_batch, epoch+1)
                    cv_g_loss, cv_d_loss = eval_one_epoch(sess, coord,
                            cv_model, cv_num_batch)
                    end = datetime.datetime.now()
                    print(("Epoch %02d: TRAIN AVG.LOSS "
                        "%.5F(G_loss, lrate(%e)), %.5F(D_loss, lrate(%e)), "
                        "CROSSVAL AVG.LOSS "
                        "%.5F(G_loss), %.5F(D_loss), TIME USED: %.2fmin") % (
                            epoch+1, tr_g_loss, tr_model.g_learning_rate,
                            tr_d_loss, tr_model.d_learning_rate,
                            cv_g_loss, cv_d_loss, (end-start).seconds/60.0))
                    sys.stdout.flush()
                    FLAGS.d_learning_rate *= FLAGS.halving_factor
                    FLAGS.g_learning_rate *= FLAGS.halving_factor
                    tr_model.d_learning_rate, tr_model.g_learning_rate = \
                        FLAGS.d_learning_rate, FLAGS.g_learning_rate
                    tr_model.save(tr_model.save_dir, epoch+1)
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
        action='store_true',
        help="Flag indicating decoding or training."
    )
    parser.add_argument(
        "--tr_list_file",
        type=str,
        required=True,
        help="TFRecords train set list file."
    )
    parser.add_argument(
        "--cv_list_file",
        type=str,
        required=True,
        help="TFRecords validation set list file."
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
        default=0.001,
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
        default=5,
        help="Min number of epoches to run trainer without halving."
    )
    parser.add_argument(
	"--max_epoches",
        type=int,
        default=30,
        help="Max number of epoches to run trainer totally."
    )
    parser.add_argument(
        "--halving_factor",
        type=float,
        default=0.8,
        help="Factor for halving."
    )
    parser.add_argument(
        "--start_halving_impr",
        type=float,
        default=0.0,
        help="Halving when ralative loss is lower than start_halving_impr."
    )
    parser.add_argument(
        "--end_halving_impr",
        type=float,
        default=0.01,
        help="Stop when relative loss is lower than end_halving_impr."
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=12,
        help='The num of threads to read tfrecords files.'
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="exp/segan",
        help="Directory to put the train result."
    )
    parser.add_argument(
        "--denoise_epoch",
        type=int,
        default=5,
        help="Epoch where noise in disc is removed."
    )
    parser.add_argument(
        "--l1_remove_epoch",
        type=int,
        default=30,
        help="Epoch where L1 in G is removed."
    )
    parser.add_argument(
        "--bias_deconv",
        type=bool,
        default=True,
        help="Flag to specify if we bias deconvs."
    )
    parser.add_argument(
        "--bias_downconv",
        type=bool,
        default=True,
        help="Flag to specify if we bias downconvs."
    )
    parser.add_argument(
        "--bias_D_conv",
        type=bool,
        default=True,
        help="Flag to specify if we bias D_convs."
    )
    parser.add_argument(
        "--init_l1_weight",
        type=float,
        default=100.0,
        help="Init L1 lambda."
    )
    parser.add_argument(
        "--g_nl",
        type=str,
        default="leaky",
        help="Type of nonlinearity in G: leaky or prelu."
    )
    parser.add_argument(
        "--deconv_type",
        type=str,
        default="deconv",
        help="Type of deconv method: deconv or nn_deconv"
    )
    parser.add_argument(
        "--beta_1",
        type=float,
        default=0.5,
        help="Adam beta 1."
    )
    parser.add_argument(
        "--init_noise_std",
        type=float,
        default=0.5,
        help="Init noise std"
    )
    parser.add_argument(
        "--g_type",
        type=str,
        default="ae",
        help="Type of G to use: ae or dwave."
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
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
