import pandas as pd
import random
import os
import numpy as np
import re
#import librosa

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# read_csv function which is used to read the required CSV file
#data = pd.read_csv('/kuacc/users/bbiner21/input.csv') 

csv_path = '/kuacc/users/bbiner21/input.csv'
dataset_dir = '/kuacc/users/bbiner21/ast/egs/custom/data'
ncd_csv_path = '/kuacc/users/bbiner21/ast/egs/custom/nocall_detection_probabilities.csv'
#ncd_csv_path = '/kuacc/users/bbiner21/ast/egs/custom/call_detection_output_small_dataset.csv'

data = pd.read_csv(csv_path) 
ncd_data = pd.read_csv(ncd_csv_path)

print("Initial data shape: ")
print(data.shape)

data = data.drop(["type","scientific_name","common_name","author","date","license","url"], axis=1)



valid_audio_list = get_immediate_files(dataset_dir+'/custom_data/audio_16k')

for index, row in data.iterrows(): 
    if row['filename'] not in valid_audio_list:
        data = data.drop(labels=index, axis=0)
        

print("after removing some rows and columns")
print(data.shape)

labels = (data['primary_label'] ).unique()
print(len(labels))


sec_labels = [re.sub('[\[\'\]]', '', x) for x in data['secondary_labels']]
sec_labels = [(re.sub(r"\s+", "", x)).split(',') for x in sec_labels]

for seq in sec_labels: # this is needed because of the DStore error i dont have all thefiles hence missing labels
    for element in seq:
        if element not in labels and element != '':
            labels = np.append(labels,element)
            
labels = np.append(labels,'nocall')    
print(len(labels))


target_dict = {}
mid_str = 'a'
mid_values = []

for ind, label in enumerate(labels):
    target_dict[label] = str(ind)
    mid_values.append(mid_str + str(ind).zfill(3))

    
ind_values = list(range(len(labels)))

custom_dict = {'index': ind_values, 'mid':mid_values, 'display_name':labels }
df = pd.DataFrame(data=custom_dict)
#df.to_csv(dataset_dir + '/custom_labels_one_stage.csv') # custom_labels part completed 


targets = []
call_binaries = []
maxLen = 14
# instead of using pre calculated no call detector from last years winner we are going to create our own
ncd_dict =  {}
ncd_binaries = []

for index, row in ncd_data.iterrows():
    prev_segments = ncd_dict.get(row['filenames'],'')
    if prev_segments == '':
        ncd_dict[row['filenames']] =  str(row['birdcall'])
    else:
        ncd_dict[row['filenames']] = prev_segments + ' ' + str(row['birdcall'])




for index, row in data.iterrows():
    curr_targets = [target_dict[row['primary_label']]]
    if row['secondary_labels'] != '[]':
        temp_secondary = (re.sub('[\[\'\]]', '', row['secondary_labels']))
        temp_secondary = (re.sub(r"\s+", "", temp_secondary)).split(',')

        secondary_targets = [target_dict[x] for x in temp_secondary]
        curr_targets += secondary_targets
    
    
    #curr_targets = [str(x) for x in curr_targets]
    targets.append("-".join(curr_targets))
    
    ncd_binaries.append(ncd_dict[row['filename']])
    
    
    
    #file_path = dataset_dir +'/custom_data/audio_16k/' + row['filename']
    #y, sr = librosa.load(path = file_path , sr = 16000)
    #durations.append(librosa.get_duration(y=y, sr=sr))

#     if index % 10000 == 0:
#         print(index)


    
    
#     probs = row['nocalldetection'].split()
#     if len(probs) >= 10:
#         probs = probs[0:10]
#     probs_float = [float(x) for x in probs]
#     arr = np.array(probs_float)
#     arr_bool = (arr>0.5).astype(float)
#     if len(arr_bool) < 10:
#         arr_bool = np.append(arr_bool,np.zeros(maxLen - len(arr_bool),dtype=int)) 
#     call_binaries.append(" ".join(str(x) for x in arr_bool))
    
num_fold =5
sample_per_fold = len(data)//num_fold
remainder = len(data) % num_fold

folds = []
for i in range(num_fold):
    for j in range(sample_per_fold):
        folds.append(i+1)

for j in range(remainder):
    folds.append(j+1)

random.shuffle(folds)
#
#data = data.drop(["nocalldetection"],axis=1)


data['fold'] = folds
data['target'] = targets
#data['durations'] = durations
#data["call_detection"] = call_binaries

data['call_detection'] = ncd_binaries
data = data[["filename","fold","target","primary_label",'call_detection', 'latitude','longitude']]

data.to_csv(dataset_dir + '/custom_data/custom_meta_duration_probs.csv')