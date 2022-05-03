# modified from:
# Author: Yuan Gong

import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random

import sys

import matplotlib
import matplotlib.pyplot as plt
import time

from pathlib import Path

import pandas as pd
import librosa
from tqdm import tqdm
pd.options.display.max_columns = 100

from skimage.transform import rescale, resize, downscale_local_mean


import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import soundfile as sf
import tracemalloc

PERIOD = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 576

POSWEIGHT=10
SR=32000



def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        
        self.data = data_json['data']
        
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        #700-128

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda
# modified version of _wav2fbank to include more segments from a single audio file
    def _wav2fbank2(self, filename, segment):
        
        new_sr = 16000
        waveform, sr = torchaudio.load(filename)
        
        #transform = torchaudio.transforms.Resample(sr, new_sr)
        #waveform = transform(waveform)
        
        waveform = waveform - waveform.mean()
        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        #fbank = torch.load(filename)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        
        
        
        
        start_ind = (target_length - 12)*segment
        end_ind = (target_length - 12)*(segment+1) + 12 # taking next 12 because 512- 500
        p = end_ind - n_frames
        
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
            fbank = fbank[start_ind:end_ind, :]
        elif p < 0:
            fbank = fbank[start_ind:end_ind, :]

        #mask_200 = torch.nn.ZeroPad2d((0, 0, 0, 200)) # 700- 500 
        #fbank = mask_200(fbank)

            
            
        return fbank, 0
    
    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        #tracemalloc.start()
        #snapshot1 = tracemalloc.take_snapshot()
        datum = self.data[index]

        label_indices = np.zeros(self.label_num)
        label_weights = np.full(self.label_num,0.05)
        

        
        fbank, mix_lambda = self._wav2fbank2(datum['wav'],datum['segment'])
        for index,label_str in enumerate (datum['labels'].split('-')):
            label_indices[int(self.index_dict[label_str])] = 1.0
            label_weights[int(self.index_dict[label_str])] = 1.0

        sec_labels = datum['sec_labels']


        for sec_label in sec_labels:
             label_weights[int(self.index_dict[sec_label])] = 0.0


        
        #print(type(label_indices))
#             if index == 0:
#                 label_indices[int(self.index_dict[label_str])] = datum['prob']
#             else:
#                 label_indices[int(self.index_dict[label_str])] = datum['prob']*0.6


        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
#         if index == 0:
#             plot_mel_fbank(fbank)

        #metadata = torch.tensor([datum['latitude'], datum['longitude']])
    
#         filename = datum['wav'].split('/')[-1]
#         segment = datum['segment']

        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #print("[ Top 10 differences ]")
        #for stat in top_stats[:10]:
         #   print(stat)
        if fbank.shape[0] != 512:  #todo vmlocate to discard or debug, only a few segments are problematic for some reason
            #print(datum['wav'],0)
            return torch.zeros(512, fbank.shape[1]), label_indices, label_weights 

        return fbank, label_indices, label_weights

    def __len__(self):
        return len(self.data)