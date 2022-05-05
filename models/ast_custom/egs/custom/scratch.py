import pandas as pd
import random
import os
import numpy as np
import re
import librosa


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# read_csv function which is used to read the required CSV file
#data = pd.read_csv('/kuacc/users/bbiner21/input.csv') 

csv_dir = '../../../../dataset/metadata/meta_sm.csv'
dataset_dir = '/datasets/xeno_canto/sm_dataset/'
base_path = './data'
data = pd.read_csv(csv_dir) 
print("Initial data shape: ")
print(data.shape)


data = data.drop(["type", "latitude","longitude","scientific_name","common_name","url"], axis=1)



valid_audio_list = get_immediate_files(dataset_dir)



for index, row in data.iterrows():
    if row['filename'] not in valid_audio_list:
        data = data.drop(labels=index, axis=0)
        

print("after removing some rows and columns")
print(data.shape)

labels = (data['primary_label'] ).unique()
print(len(labels))


#sec_labels = [re.sub('[\[\'\]]', '', x) for x in data['secondary_label']]
#sec_labels = [(re.sub(r"\s+", "", x)).split(',') for x in sec_labels]

#for seq in sec_labels: # this is need because of the DStore error i dont have all thefiles hence missing labels
#    for element in seq:
#        if element not in labels and element != '':
#labels = np.append(labels,element)


#print(len(labels))
#print(labels)

target_dict = {}
mid_str = 'a'
mid_values = []

for ind, label in enumerate(labels):
    target_dict[label] = str(ind)
    mid_values.append(mid_str + str(ind).zfill(3))
  
ind_values = list(range(len(labels)))

custom_dict = {'index': ind_values, 'mid':mid_values, 'display_name':labels }
df = pd.DataFrame(data=custom_dict)
df.to_csv(base_path + '/custom_labels.csv')




targets = []
call_binaries = []
maxLen = 10
durations = []


for index, row in data.iterrows():
    curr_targets = [target_dict[row['primary_label']]][0]
    #if row['secondary_label'] != '[]':
     #   temp_secondary = (re.sub('[\[\'\]]', '', row['secondary_label']))
      #  temp_secondary = (re.sub(r"\s+", "", temp_secondary)).split(',')

       # secondary_targets = [target_dict[x] for x in temp_secondary]
       # curr_targets += secondary_targets
    
    #curr_targets = [str(x) for x in curr_targets]
    #targets.append("-".join(curr_targets))
    targets.append(curr_targets)
    file_path = dataset_dir + row['filename']
    y, sr = librosa.load(path = file_path , sr = 16000)
    durations.append(librosa.get_duration(y=y, sr=sr))

    if index % 10000 == 0:
        print(index)

    # instead of using pre calculated no call detector from last years winner we are going to create our own
    
    
#     probs = row['nocalldetection'].split()
#     if len(probs) >= 10:
#         probs = probs[0:10]
#     probs_float = [float(x) for x in probs]
#     arr = np.array(probs_float)
#     arr_bool = (arr>0.5).astype(float)
#     if len(arr_bool) < 10:
#         arr_bool = np.append(arr_bool,np.zeros(maxLen - len(arr_bool),dtype=int)) 
#     call_binaries.append(" ".join(str(x) for x in arr_bool))
    
num_fold =2
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
data['durations'] = durations
#data["call_detection"] = call_binaries
data = data[["filename","fold","target","primary_label","durations"]]

data.to_csv(base_path + '/custom_data/custom_meta.csv')