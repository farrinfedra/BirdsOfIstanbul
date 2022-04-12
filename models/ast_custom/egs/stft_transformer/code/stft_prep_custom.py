# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import json
import os
import zipfile
import wget
import sys 
import pandas as pd
# label = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_label.csv', delimiter=',', dtype='str')
# f = open("/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices.csv", "w")
# f.write("index,mid,display_name\n")
#
# label_set = []
# idx = 0
# for j in range(0, 5):
#     for i in range(0, 10):
#         cur_label = label[i][j]
#         cur_label = cur_label.split(' ')
#         cur_label = "_".join(cur_label)
#         cur_label = cur_label.lower()
#         label_set.append(cur_label)
#         f.write(str(idx)+',/m/07rwj'+str(idx).zfill(2)+',\"'+cur_label+'\"\n')
#         idx += 1
# f.close()
#

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
# if os.path.exists('./data/custom_data/audio_16k') == False:
#     #esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
#     #wget.download(esc50_url, out='./data/')
#     #with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
#     #   zip_ref.extractall('./data/')
#     #os.remove('./data/ESC-50-master.zip')
     
#         # Lets say we manually upload code for now
#         #  put data to ./data/custom_data/audio
        
#     dataset_dir = '/datasets/birdclef-2021/train_short_audio'
#     base_dir = './data/custom_data'
#     os.mkdir('./data/custom_data/audio_16k/')
#     for name in os.listdir(dataset_dir):
#         curr_dir = os.path.join(dataset_dir, name)
    
#     # convert the audio to 16kHz

#         audio_list = get_immediate_files(curr_dir)
#         for audio in audio_list:
#             print('sox ' + curr_dir + '/' + audio + ' -r 16000 ' + base_dir + '/soundscape_audio_16k/' + audio)
#             os.system('sox ' + curr_dir + '/' + audio + ' -r 16000 ' + base_dir + '/soundscape_audio_16k/' + audio)

#sys.exit()
            
#label_set = np.loadtxt('./data/custom_labels_one_stage.csv', delimiter=',', dtype='str')

# fix bug: generate an empty directory to save json files
# if os.path.exists('./data/datafiles') == False:
#     os.mkdir('./data/datafiles')
    
datafiles_dir = '../datafiles/stft_datafiles'

if os.path.exists(datafiles_dir) == False:
    os.mkdir(datafiles_dir)

#nocall_label = 'a398'
    
df = pd.read_csv("../input/stft_meta.csv")
meta = df.values
#meta = np.genfromtxt('../input/stft_meta.csv', delimiter=',', dtype='str')

for fold in [1]: # [1,2,3,4,5]
    base_path = "../data"
    #base_path = './data/fbank_tensors/'
    base_for_norm = "/kuacc/users/bbiner21/ast/egs/stft_transformer"
    
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):   # range(0, len(meta)) 200 was small dataset
        cur_label = meta[i][2]
        cur_path = meta[i][4] # '../data/' + 
        cur_fold = int(meta[i][7])
        
        cur_dict = {"wav": cur_path, "labels": cur_label, 'weight': meta[i][6], 'sec_labels':meta[i][3], 'class':meta[i][5],'length':meta[i][1]}
        
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)
               
        
    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open(datafiles_dir + '/stft_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open(datafiles_dir + '/stft_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)
        
print('Finished custom Preparation')