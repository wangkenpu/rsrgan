#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail

stage=1

nj=3
val_size=100
train_dir=data/train/train_test
tr_list=$train_dir/tr.list
cv_list=$train_dir/cv.list

# Data prepare
if [ $stage -le 0 ]; then
  echo "======================================================================"
  echo "                      Prepare tr and cv data                          "
  echo "======================================================================"

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
  logdir=exp
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
  echo "Make TFRecords files sucessed."
  echo "========================== Data prepare done ========================="
  echo ""
fi


# Train model
if [ $stage -le 1 ]; then
  CUDA_VISIBLE_DEVICES="2,3" TF_CPP_MIN_LOG_LEVEL=2 \
    python scripts/train_segan.py \
      --tr_list_file=$tr_list \
      --cv_list_file=$cv_list \
      --save_dir=exp/segan \
      --input_dim=257 \
      --output_dim=40 \
      --left_context=5 \
      --right_context=5 \
      --batch_size=256 \
      --g_learning_rate=0.001 \
      --d_learning_rate=0.001 \
      --min_epoches=2 \
      --max_epoches=20 \
      --halving_factor=0.8 \
      --start_halving_impr=0.0 \
      --end_halving_impr=0.01 \
      --num_threads=10 \
      --denoise_epoch=5 \
      --l1_remove_epoch=25 \
      --bias_deconv=True \
      --bias_downconv=True \
      --bias_D_conv=True \
      --init_l1_weight=100.0 \
      --g_nl="prelu" \
      --deconv_type="deconv" \
      --beta_1=0.5 \
      --init_noise_std=0.5 \
      --g_type="ae" \
      --num_gpu=2 || exit 1;
fi


echo "========================================================================"
echo "Finished successfully on $(date)"
echo "========================================================================"

exit 0
