#!/bin/bash

# Copyright 2017  Ke Wang

set -euo pipefail
[ -f path.sh ] && . ./path.sh

stage=0

# Begin configuration section
nj=10
cmd=run.pl
# End configuration section

rir_list=data/train/rir_list
noise_list=data/train/noise_list
rvb_dir=/home/train06/wangke/data/xiaomi-asr-data-001-wav-1of4-rvb/wavs
input=data/train
output=data/train_rvb
logdir=exp/rvb

mkdir -p $logdir || exit 1;
mkdir -p $rvb_dir || exit 1;

if [ $stage -le 0 ]; then
  python steps/data/reverberate_bash.py \
    --rir-set-parameters ${rir_list} \
    --reverberation-wav-dir ${rvb_dir} \
    --num-replications 1 \
    --foreground-snrs-lower-bound 5 \
    --foreground-snrs-upper-bound 20 \
    --background-snrs-lower-bound 5 \
    --background-snrs-upper-bound 20 \
    --prefix rvb \
    --speech-rvb-probability 1.0 \
    --pointsource-noise-addition-probability 1.0 \
    --isotropic-noise-addition-probability 1.0 \
    --rir-smoothing-weight 0.3 \
    --noise-smoothing-weight 0.3 \
    --max_noises_adding 1 \
    --random-seed 1 \
    --shift-output true \
    --source-sampling-rate 16000 \
    --include-original-data false \
    --normalize-output true \
    --verbose 1 \
    $input $output
    #--noise-set-parameters ${noise_list} \
  echo "Preparing adding reverberation and noise commands sucessfully."
fi

if [ $stage -le 1 ]; then
  split_command=""
  for n in $(seq $nj); do
    split_command="$split_command $logdir/command.$n.sh"
  done

  utils/split_scp.pl $output/addNoise.sh $split_command || exit 1;

  $cmd JOB=1:$nj $logdir/add_noise.JOB.log \
    bash $logdir/command.JOB.sh || exit 1;

  echo "Sucessed corrupting the wave files."
fi
