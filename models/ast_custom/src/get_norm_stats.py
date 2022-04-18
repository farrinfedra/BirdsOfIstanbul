# modified from:
# Author: Yuan Gong

# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np
import dataloader_custom

# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 24, 'timem': 96, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'custom'}

data_loader = torch.utils.data.DataLoader(
    dataloader_custom.AudiosetDataset('../egs/custom/data_nocall/datafiles/custom_train_data_1.json', label_csv='../egs/custom/data_nocall/custom_labels.csv',
                                audio_conf=audio_conf), batch_size=96, shuffle=False, num_workers=8, pin_memory=True)



mean=[]
std=[]

for i, (audio_input, labels) in enumerate(data_loader):
    print(audio_input)
    print(labels)
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    if i % 100 == 0:
        print(np.mean(mean), np.mean(std))
    break
        
print(np.mean(mean), np.mean(std))