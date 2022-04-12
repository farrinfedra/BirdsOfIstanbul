import os
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from pqdm.processes import pqdm

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

dataset_dir = '/datasets/freefield1010/'
valid_freefield_dict = {}
for folder_name in ['01','02','03','04','05','06','07','08','09','10']:
    file_list = get_immediate_files(dataset_dir + folder_name)
    for file in file_list:
        valid_freefield_dict[file] = folder_name

     
    
train_ff1010 = pd.read_csv('../input/ff1010bird_metadata.csv')
train_ff1010

train_ff1010 = train_ff1010.sort_values(by='itemid').reset_index(drop=True)
train_ff1010

def get_clip(itemid):
    data_path = Path('/datasets/freefield1010')
    folder_name = valid_freefield_dict[str(itemid) + '.wav']
    
    
    path = data_path / folder_name / ('%d.wav' % itemid)
    clip, sr_native = librosa.load(path, sr=None, mono=True, dtype=np.float32)
    sr = 32000
    if sr_native != 0:
        clip = librosa.resample(clip, sr_native, sr, res_type='kaiser_best')
    else:
        print('null sr_native')
    return clip, sr, sr_native

train_ff1010 = train_ff1010[train_ff1010.hasbird == 0].reset_index(drop=True)
train_ff1010

def work_sub(itemid):
    output_path = Path('../data/')
    clip, sr, sr_native = get_clip(itemid)
    clip = clip.astype('float32')
    length = clip.shape[0] 
    filename = 'ff1010_%d_0.npy' % (itemid)
    np.save(output_path / filename, clip)
    
    return sr, sr_native, length


res = pqdm(train_ff1010.itemid, work_sub, n_jobs=8)

train_ff1010['primary_label'] = ''
train_ff1010['secondary_labels'] = None
train_ff1010['sr'] = [r[0] for r in res]
train_ff1010['sr_native'] = [r[1] for r in res]
train_ff1010['length'] = [r[2] for r in res]
train_ff1010['duration'] = train_ff1010['length'] / 32000
train_ff1010['filename'] = ['ff1010_%d_0.npy' % (itemid) for itemid in train_ff1010['itemid']]
train_ff1010

train_ff1010['secondary_labels'] = [[]] * len(train_ff1010)

columns = ['duration', 'length', 'primary_label', 'secondary_labels', 'filename']
train_ff1010[columns].to_csv('../input/train_ff1010.csv', index=False)        


