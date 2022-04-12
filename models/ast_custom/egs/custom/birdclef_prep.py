import numpy as np
import json
import os
import zipfile
import wget
import sys 
import pandas as pd


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


datafiles_dir = './data/datafiles_bc'

if os.path.exists(datafiles_dir) == False:
    os.mkdir(datafiles_dir)
    
df = pd.read_csv('./small_meta.csv')
meta = df.values

train_wav_list = []

for i in range(0, len(meta)): 
    cur_label = meta[i][1]
    cur_trg = 'a' + str(meta[i][4]).zfill(3)
    fname = meta[i][3].split('/')[-1]
    cur_path = './birdclef_temp_data/' + cur_label + '/' + fname
    
    cur_dict = {"wav": cur_path, "labels": cur_trg}
    train_wav_list.append(cur_dict)
    
with open(datafiles_dir + '/small_train_data_' + '.json', 'w') as f:
    json.dump({'data': train_wav_list}, f, indent=1)