#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail

stage=1

nj=30
val_size=3000
#train_dir=data/train/train_test
#train_dir=data/train/train_100h
train_dir=data/train/train_100h_new_mfcc
#train_dir=data/train/train_390h
#test_dir=data/test/test001_mfcc
test_dir=data/test/test001-3000-smallroom-mfcc
logdir=exp
tr_list=$train_dir/tr.list
cv_list=$train_dir/cv.list
test_list=$test_dir/test.list


# Data prepare
if [ $stage -le 0 ]; then
  echo "Prepare tr and cv data"

  # Prepare Numpy formatCMVN file
  echo "Make Numpy format Global CMVN file ..."
  python io_funcs/convert_cmvn_to_numpy.py \
    --inputs=$train_dir/inputs.cmvn \
    --labels=$train_dir/labels.cmvn \
    --save_dir=$train_dir
  echo "Make CMVN done."

  # Split tr & cv sets
  echo "Split tr and cv sets ..."
  python scripts/get_train_val_scp.py \
     --val_size=$val_size \
     --data_dir=$train_dir
  echo "Split done."

  # Make TFRecords file
  echo "Begin making TFRecords files ..."
  if [ ! -d $logdir ]; then
    mkdir -p $logdir || exit 1;
  fi
  if [ -f $logdir/.cv.error ]; then
    rm -rf $logdir/.cv.error || exit 1;
  elif [ -f $logdir/.tr.error ]; then
    rm -rf $logdir/.tr.error || exit 1;
  fi

  # cv set
  declare -i verbose=30
  [ -d $train_dir/tfrecords ] && (rm -rf $train_dir/tfrecords || exit 1;)
  mkdir -p $train_dir/tfrecords || exit 1;
  TF_CPP_MIN_LOG_LEVEL=1 python io_funcs/make_tfrecords.py \
    --verbose=$verbose \
    --inputs=$train_dir/cv/inputs.scp \
    --labels=$train_dir/cv/labels.scp \
    --cmvn_dir=$train_dir \
    --apply_cmvn=True \
    --output_dir=$train_dir/tfrecords \
    --name="cv" || touch $logdir/.cv.error &
  echo "$train_dir/tfrecords/cv.tfrecords" > $cv_list

  # tr set
  bash scripts/split_scp.sh --nj $nj $train_dir/tr
  [ -f $tr_list ] && (rm -rf $tr_list || exit 1);
  for i in $(seq $nj); do
  (
    TF_CPP_MIN_LOG_LEVEL=1 python io_funcs/make_tfrecords.py \
      --verbose=$verbose \
      --inputs=$train_dir/tr/split${nj}/inputs${i}.scp \
      --labels=$train_dir/tr/split${nj}/labels${i}.scp \
      --cmvn_dir=$train_dir \
      --apply_cmvn=True \
      --output_dir=$train_dir/tfrecords \
      --name="tr${i}"
    echo "$train_dir/tfrecords/tr${i}.tfrecords" >> $tr_list
  ) || touch $logdir/.tr.error &
  done
  wait

  if [ -f $logdir/.tr.error ] || [ -f $logdir/.cv.error ]; then
    echo "$0: there was a problem while making TFRecords" && exit 1
  fi
  [ -f $train_dir/batch_num.txt ] && rm $train_dir/batch_num.txt
  echo "Make train TFRecords files sucessed."
  echo ""
fi


if [ $stage -le 1 ]; then
  echo "Prepare test data"
  if [ -f $logdir/.test.error ]; then
    rm -rf $logdir/.test.error || exit 1;
  fi
  declare -i verbose=30
  [ -d $test_dir/tfrecords ] && (rm -rf $test_dir/tfrecords || exit 1;)
  mkdir -p $test_dir/tfrecords || exit 1;
  TF_CPP_MIN_LOG_LEVEL=1 python io_funcs/make_tfrecords.py \
    --test \
    --verbose=$verbose \
    --inputs=$test_dir/test.scp \
    --cmvn_dir=$train_dir \
    --apply_cmvn=True \
    --output_dir=$test_dir/tfrecords \
    --name="test" || touch $logdir/.test.error &
  echo "$test_dir/tfrecords/test.tfrecords" > $test_list
  wait

  if [ -f $logdir/.test.error ]; then
    echo "$0: there was a problem while making TFRecords" && exit 1
  fi
  echo "Make test TFRecords files sucessed."
  echo ""
fi
exit 0


# Train model
if [ $stage -le 2 ]; then
  echo "$(date): $(hostname)"
  CUDA_VISIBLE_DEVICES="0,1,2,3" TF_CPP_MIN_LOG_LEVEL=2 \
    python scripts/train_rnn.py \
      --data_dir=$train_dir \
      --tr_list_file=$tr_list \
      --cv_list_file=$cv_list \
      --g_type="lstm" \
      --save_dir=exp/1208_blstm_test \
      --batch_size=8 \
      --g_learning_rate=0.0003 \
      --batch_norm=False \
      --keep_prob=1.0 \
      --l2_scale=0.0 \
      --input_dim=40 \
      --output_dim=40 \
      --left_context=0 \
      --right_context=0 \
      --min_epochs=20 \
      --max_epochs=25 \
      --end_improve=0.001 \
      --num_threads=32 \
      --num_gpu=4 || exit 1;

  echo "Finished training successfully on $(date)"
  echo ""
fi


# Decode
if [ $stage -le 3 ]; then
  echo "Start decoding test data"
  TF_CPP_MIN_LOG_LEVEL=2 python scripts/train_rnn.py \
      --decode \
      --data_dir=$train_dir \
      --test_list_file=$test_list \
      --g_type="res_lstm_l" \
      --save_dir=exp/1020_res_lstm_l \
      --batch_norm=False \
      --input_dim=257 \
      --output_dim=40 \
      --left_context=0 \
      --right_context=0 \
      --batch_size=1 \
      --keep_prob=1.0 \
      --l2_scale=0.0 \
      --num_threads=30 || exit 1;
  echo "Decoding done"
fi

exit 0
