import sys
import timm

import json
import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from timm.models.layers import to_2tuple,trunc_normal_
import matplotlib.pyplot as plt
import torchaudio
import csv

import tensorflow.compat.v1 as tf
import distutils.util
import tensorflow_io as tfio
from numba import cuda 
import tensorflow as tf
import IPython 

test_audio_dir = '/datasets/xeno_canto/sm_dataset/'

file_list = os.listdir(test_audio_dir)
if '' in file_list:
    file_list.remove('')
print('Number of test soundscapes:', len(file_list))

with open('../egs/custom/data_nocall/custom_labels.csv') as sbfile:
    df = pd.read_csv(sbfile)
    birds = df['display_name'].tolist()
    

def make_index_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['display_name']] = int(row['index'])
            line_count += 1
    return name_lookup


class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=384, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=399, fstride=10, tstride=10, input_fdim=128, input_tdim=512, imagenet_pretrain=True, audioset_pretrain=True, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
        #self.v = audio_model.module.v
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        
        #self.v.patch_embed = patch_emb
        self.v.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)
            

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=512):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
  
    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x

def init_model():    
    print("MODEL IS INITIALIZED\n")
    audio_model = ASTModel(label_dim=2, fstride=10, tstride=10, input_fdim=128,
                                      input_tdim=512, imagenet_pretrain=True,
                                      audioset_pretrain=True, model_size='base384')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#path = '/kuacc/users/fsofian19/ast_custom_stage1/src/models/best_audio_model_5s.pth' 
#path = '/kuacc/users/fsofian19/ast_custom_stage1/egs/custom/exp_2021/test-custom-5s-Interval-f10-t10-impTrue-aspTrue-b36-lr1e-5/fold5/models/best_audio_model.pth'
    torch.cuda.empty_cache()
#path = '/kuacc/users/fsofian19/ast_custom_stage1/egs/custom/exp_2021/test-custom-5s-Interval-f10-t10-impTrue-aspTrue-b36-lr1e-5/fold5/models/best_audio_model.pth'
    path = '../pretrained_models/best_audio_model_5s.pth'
    sd = torch.load(path, map_location=device)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.load_state_dict(sd,strict=False)
    return audio_model




def test(audio_model):
    index_dict = make_index_dict('../egs/custom/data_nocall/custom_labels.csv')
    # norm_mean = -6.0138397
    # norm_std = 4.589279 
    print("STARTING TESTING FOR NOCALL PROBS\n")

    index_dict = make_index_dict('../egs/custom/data_nocall/custom_labels.csv')
    norm_mean, norm_std = -5.5194726, 4.5720654

    # This is where we will store our results
    pred = {'file': [], 'target': []}

    # Process audio files and make predictions
    for afile in file_list:
    # Complete file path
        path = test_audio_dir + afile
    #path = '../input/birdclef-2022/train_audio/akiapo/XC122399.ogg'
        new_sr = 16000
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)  # converting multiple channels to mono

        transform = torchaudio.transforms.Resample(sr, new_sr)
        waveform = transform(waveform)
    
    
        n_seconds = waveform.shape[1]/new_sr
        #print(n_seconds)

        chunk_num = int(n_seconds//5)
        if waveform.shape[1]%new_sr > 0.0:
            chunk_num += 1

        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=new_sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        # Let's assume we have a list of 12 audio chunks (1min / 5s == 12 segments)
        chunks = [[] for i in range(chunk_num)]
        #print(chunk_num)
        # Make prediction for each chunk
        # Each scored bird gets a random value in our case
        # since we don't actually have a model
        n_frames = fbank.shape[0]
        #print(n_frames)
        temp = ''
        for i in range(chunk_num):
            #debug_memory()
            if i != 0:
                temp = temp + ' '
            chunk_start_time = i*5
            chunk_end_time = (i + 1) * 5

            start = 500*i
            end = 500*(i+1) +12

            #print(n_frames)

            if end > n_frames:
                m = torch.nn.ZeroPad2d((0, 0, 0, int(end-n_frames)))            
                fbank = m(fbank)

            cur_fbank = fbank[start:end,:]
        #print(torch.mean(cur_fbank))
        #temp = torch.nn.ZeroPad2d((0, 0, 0, 12))            
        #cur_fbank = temp(cur_fbank)
            cur_fbank = (cur_fbank - norm_mean) / (norm_std * 2)

            cur_fbank = cur_fbank.unsqueeze(0)
            out = audio_model(cur_fbank)
            out = torch.sigmoid(out)
            out = out.to('cpu').detach()
        #print(out)
            ind = index_dict['bird']
            score = out[0,ind]
        # Assemble the row_id which we need to do for each scored bird
            row_id = afile + '_' + str(chunk_end_time)
        # Put the result into our prediction dict and
        # apply a "confidence" threshold of 0.5
            temp = temp + ('1' if score > 0.5 else '0')
        
        #temp.append(1 if score > 0.5 else 0)
        pred['file'].append(afile)
        pred['target'].append(temp)
    return pred

def create_call_file(pred):
    print("CREATING CALL PROBABILTIES FILE\n")
    df = pd.DataFrame(pred, columns = ['file', 'target'])
    df.to_csv("call_probs.csv", index=False) 
        

        
if __name__ == '__main__':
    model = init_model()
    pred = test(model)
    create_call_file(pred)