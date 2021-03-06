# Robust Speech Recognition Using Generative Adversarial Networks (GAN)

## Introduction

This is the repository of the RSRGAN project. Our original paper can be found [here](https://arxiv.org/abs/1803.10132).

In this work we investigate the use of generative adversarial networks (GANs) in speech dereverberation for robust speech recognition.

Our RIRs were from [here](http://www.openslr.org/28/), and we used KALDI to simulate reverberant speech.

All the project was developed with TensorFlow and KALDI and our project was based on [SEGAN](https://github.com/santi-pdp/segan).

## Dependencies

- Python 2.7
- TensorFlow 1.4.0
- KALDI

## Date preparation and GAN training

1. You should prepare your data according KALDI format as following:
   ```shell
   data
       |- wav.scp
       |- ...
   ```
   then, runing the following command to simulate reverbrant data.
   ```shell
   # remember to change the data path in run.sh
   bash reverberate/run.sh
   ```

2. After this step, you can get reverbrant waves and their counterparts, i.e., the clean speeches. Using KALDI command ```compute-spectrogram-feats``` to get log-power spectrum (LPS) features as your inputs. Using ```compute-mfcc-feats``` to get MFCC feature as your labels. In our experiments, we used 257-dim LPS and 40-dim MFCC without delta and delta-delta.
   - LPS: hamming window
   - MFCC: default config file in KALDI WSJ recipe (conf/mfcc_hires.conf)

3. Training GAN
   ```shell
   # shell file name  with 'placeholder' is the script that training 'G' and 'D' using the same min-batch's data.
   # remeber to change the data path in shell files
   # you can also try other shell files to reproduce the other experimental results. (E.g., DNN, RCED, LSTM)
   bash run_gan_rnn_placeholder.sh
   ```

## ASR decoding

1. You should have a well-trained ASR model and make sure your AM's input feature is the same with GAN generator's output feature.
2. Then you can use any kind of decoder to decode enhanced feature.

## Reference
If the code of this repository was useful for your research, please cite our work:

```
@inproceedings{wang2018investigating,
  author    = {Wang, Ke and Zhang, Junbo and Sun, Sining and Wang, Yujun and Xiang, Fei and Xie, Lei},
  booktitle = {Interspeech 2018},
  pages     = {1581--1585},
  title     = {{Investigating Generative Adversarial Networks based Speech Dereverberation for Robust Speech Recognition}},
  year      = {2018}
}
```

# Contact
e-mail: wangkenpu@gmail.com
