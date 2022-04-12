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
import csv

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


def make_mid_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['display_name']] = row['mid']
            line_count += 1
    return name_lookup

# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
# if os.path.exists('./data/custom_data/soundscape_audio_16k') == False:
#     #esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
#     #wget.download(esc50_url, out='./data/')
#     #with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
#     #   zip_ref.extractall('./data/')
#     #os.remove('./data/ESC-50-master.zip')
     
#         # Lets say we manually upload code for now
#         #  put data to ./data/custom_data/audio
        
#     dataset_dir = '/datasets/birdclef-2021/train_soundscapes'
#     base_dir = './data/custom_data'
#     os.mkdir('./data/custom_data/soundscape_audio_16k/')
# #     for name in os.listdir(dataset_dir):
#     curr_dir = dataset_dir
    
#     # convert the audio to 16kHz

#     audio_list = get_immediate_files(curr_dir)
#     for audio in audio_list:
#         print('sox ' + curr_dir + '/' + audio + ' -r 16000 ' + base_dir + '/soundscape_audio_16k/' + audio)
#         os.system('sox ' + curr_dir + '/' + audio + ' -r 16000 ' + base_dir + '/soundscape_audio_16k/' + audio)

# #sys.exit()
            
# label_set = np.loadtxt('./data/custom_labels.csv', delimiter=',', dtype='str')
# label_map = {}
# for i in range(0, len(label_set)):
#     label_map[label_set[i][3]] = label_set[i][1]
#print(label_map)


# fix bug: generate an empty directory to save json files
if os.path.exists('../datafiles/train_soundscape') == False:
    os.mkdir('../datafiles/train_soundscape')


base_path = '/datasets/birdclef-2021/train_soundscapes/'
#base_for_norm = "/kuacc/users/bbiner21/ast/egs/custom/data/custom_data/soundscape_audio_16k/"

meta = np.loadtxt('../../custom/data/custom_data/train_soundscape_labels.csv', delimiter=',', dtype='str', skiprows=1)
train_wav_list = []
eval_wav_list = []
all_data = []
#mid_dict = make_mid_dict('./data/custom_labels_one_stage.csv')
ogg_files = get_immediate_files('/datasets/birdclef-2021/train_soundscapes')
path_map = {}
count_nocall = 0

for i in range(0, len(ogg_files)):
    path_map[ogg_files[i].split('_')[0]] = ogg_files[i]

for i in range(0, len(meta)):   # range(0, len(meta)) 200 was small dataset
    cur_labels = [x for x in meta[i][4].split()]
    
    #cur_labels = [mid_dict.get(x,'dne') for x in cur_labels]
    #cur_labels = ['a'+x.zfill(3) for x in cur_labels]
    #label_map[meta[i][4]]

    #ogg_files = get_immediate_files('./data/custom_data/soundscape_audio_16k')
    cur_id = meta[i][2]
    cur_path = path_map.get(cur_id,'')
#     for ogg in ogg_files:
#         if cur_id == ogg.split('_')[0]:
#             cur_path = ogg
    
#     if cur_path == '' or 'dne' in cur_labels: # if path does not exists for id or label does not exists in dict
#         count_nocall += 1
#         continue # here we are also discarding time segments with no call since hey are not in our dict
   
    cur_segment = int(int(meta[i][3]) // 5 - 1)
    #cur_path = meta[i][1]
#     cur_path = meta[i][1]
    #cur_fold = int(meta[i][2])
    #curr_call_detection = (meta[i][5]).split() # 5 gives call_detection ones and zeros
    
#     if float(meta[i][5]) > 70 : #duration
#         num_segments = 14
#     else:
#         num_segments = int((float(meta[i][5]) // 5) + 1) # 5 seconds intervals

    cur_dict = {"wav": base_path + cur_path, "labels": " ".join(cur_labels),"segment":cur_segment}
    eval_wav_list.append(cur_dict)
#     for j in range(num_segments):
#         cur_dict = {"wav": base_path + cur_path, "labels": "-".join(cur_labels),"segment":j, 'filename': cur_path}
#         eval_wav_list.append(cur_dict)
    #all_data.append(cur_dict2)

print('{:d} test samples'.format(len(eval_wav_list)))

# with open('./data/datafiles/custom_train_data_'+ str(fold) +'.json', 'w') as f:
#     json.dump({'data': train_wav_list}, f, indent=1)

with open('../datafiles/train_soundscape/train_soundscape_nocall'+'.json', 'w') as f:
    json.dump({'data': eval_wav_list}, f, indent=1)

# with open('./data/datafiles/all_data'+ str(fold) +'.json', 'w') as f:
#     json.dump({'data': all_data}, f, indent=1)

print('Finished custom Preparation')