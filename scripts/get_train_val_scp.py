#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Get train and validation set."""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import pprint
import random
import sys


def main():
    inputs_scp = os.path.join(FLAGS.data_dir, "inputs.scp")
    labels_scp = os.path.join(FLAGS.data_dir, "labels.scp")
    if FLAGS.splite_dir is not None:
        tr_dir = os.path.join(FLAGS.splite_dir, "tr")
        cv_dir = os.path.join(FLAGS.splite_dir, "cv")
    else:
        tr_dir = os.path.join(FLAGS.data_dir, "tr")
        cv_dir = os.path.join(FLAGS.data_dir, "cv")
    tr_inputs_scp = os.path.join(tr_dir, "inputs.scp")
    tr_labels_scp = os.path.join(tr_dir, "labels.scp")
    cv_inputs_scp = os.path.join(cv_dir, "inputs.scp")
    cv_labels_scp = os.path.join(cv_dir, "labels.scp")

    print("Split to %s and %s" % (tr_dir, cv_dir))

    if not os.path.exists(tr_dir):
        os.makedirs(tr_dir)
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)

    with open(inputs_scp, 'r') as fr_inputs, \
            open(labels_scp, 'r') as fr_labels, \
            open(tr_inputs_scp, 'w') as fw_tr_inputs, \
            open(tr_labels_scp, 'w') as fw_tr_labels, \
            open(cv_inputs_scp, 'w') as fw_cv_inputs, \
            open(cv_labels_scp, 'w') as fw_cv_labels:

        lists_inputs = fr_inputs.readlines()
        lists_labels = fr_labels.readlines()
        if len(lists_inputs) != len(lists_labels):
            print("%s and %s have unequal lengths" % (inputs_scp, labels_scp))
            sys.exit(-1)
        if len(lists_inputs) <= FLAGS.val_size:
            print(("Validation size %s is bigger than inputs scp length %s."
                   " Please reduce validation size.") % (
                       FLAGS.val_size, len(lists_inputs)))

        lists = range(len(lists_inputs))
        random.shuffle(lists)
        for i in xrange(len(lists)):
            line_input = lists_inputs[i]
            line_label = lists_labels[i]
            if i < FLAGS.val_size:
                fw_cv_inputs.write(line_input)
                fw_cv_labels.write(line_label)
            else:
                fw_tr_inputs.write(line_input)
                fw_tr_labels.write(line_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Directory name of data to spliting."
             "(Note: inputs.scp and labels.scp)"
    )
    parser.add_argument(
        '--splite_dir',
        type=str,
        default=None,
        help="Directory to save splite data."
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=5000,
        help="Validation set size."
    )

    FLAGS, unparsed = parser.parse_known_args()
    # pp = pprint.PrettyPrinter()
    # pp.pprint(FLAGS.__dict__)
    main()
