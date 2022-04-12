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

input_path = '../input/'

train = pd.read_csv(input_path + 'train_001.csv')
train.head()

train_ff1010 = pd.read_csv(input_path + 'train_ff1010.csv')
train_ff1010['primary_label'] = 'nocall'
train_ff1010.head(n=8)

columns = ['length', 'primary_label', 'secondary_labels', 'filename']
#print(train.shape)
train = pd.concat((train[columns], train_ff1010[columns])).reset_index(drop=True)
#print(train.shape)

primary_labels = set(train.primary_label.unique())

secondary_labels = set([s for labels in train.secondary_labels for s in eval(labels)])


#len(primary_labels), len(secondary_labels), len(secondary_labels - primary_labels)

# res = [[label for label in eval(secondary_label) if label != 'rocpig1'] 
#                              for secondary_label in train['secondary_labels']]   #they remove the codename rocpig1 but whyy

# train['secondary_labels'] = res

BIRD_CODE = {}
INV_BIRD_CODE = {}
for i,label in enumerate(sorted(primary_labels)):
    BIRD_CODE[label] = i
    INV_BIRD_CODE[i] = label
    
NOCALL_CODE = BIRD_CODE['nocall']

train['class'] = [BIRD_CODE[label] for label in train.primary_label]
train['weight'] = train.groupby('class')['class'].transform('count')
train['weight'] = 1 / np.sqrt(train['weight'])
train['weight'] /= train['weight'].mean()
train.loc[train.primary_label == 'nocall', 'weight'] = 1

print("After stft processing data, train shape: ")
print(train.shape)

a_file = open("bird_code.pkl", "wb")
pickle.dump(bird_code, a_file)
a_file.close()

#csv_dir = '/kuacc/users/bbiner21/input.csv'
#dataset_dir = '/kuacc/users/bbiner21/ast/egs/custom/data'
dataset_dir = '../data'


# data = pd.read_csv(csv_dir) 
# print("Initial data shape: ")
# print(data.shape)

#data = data.drop(["type", "latitude","longitude","scientific_name","common_name","author","date","license","url"], axis=1)



# valid_audio_list = get_immediate_files(dataset_dir)

# for index, row in data.iterrows():
#     if row['filename'] not in valid_audio_list:
#         data = data.drop(labels=index, axis=0)
        

# print("after removing some rows")
# print(data.shape)

# labels = (data['primary_label'] ).unique()
# print(len(labels))


# sec_labels = [re.sub('[\[\'\]]', '', x) for x in data['secondary_labels']]
# sec_labels = [(re.sub(r"\s+", "", x)).split(',') for x in sec_labels]

# for seq in sec_labels: # this is need because of the DStore error i dont have all thefiles hence missing labels
#     for element in seq:
#         if element not in labels and element != '':
#             labels = np.append(labels,element)
            
# print(len(labels))

# target_dict = {}
# mid_str = 'a'
# mid_values = []

# for ind, label in enumerate(primary_labels):
#     target_dict[label] = str(ind)
#     mid_values.append(mid_str + str(ind).zfill(3))

    
# ind_values = list(range(len(primary_labels)))

# custom_dict = {'index': ind_values, 'mid':mid_values, 'display_name':primary_labels }
# df = pd.DataFrame(data=custom_dict)
# df.to_csv(dataset_dir + '/custom_labels.csv')






#targets = []
# call_binaries = []
# maxLen = 10
# durations = []


# for index, row in data.iterrows():
#     curr_targets = [target_dict[row['primary_label']]]
#     if row['secondary_labels'] != '[]':
#         temp_secondary = (re.sub('[\[\'\]]', '', row['secondary_labels']))
#         temp_secondary = (re.sub(r"\s+", "", temp_secondary)).split(',')

#         secondary_targets = [target_dict[x] for x in temp_secondary]
#         curr_targets += secondary_targets
    
#     #curr_targets = [str(x) for x in curr_targets]
#     targets.append("-".join(curr_targets))
#     file_path = dataset_dir +'/custom_data/audio_16k/' + row['filename']
#     y, sr = librosa.load(path = file_path , sr = 16000)
#     durations.append(librosa.get_duration(y=y, sr=sr))

#     if index % 10000 == 0:
#         print(index)
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
    
num_fold =5
sample_per_fold = len(train)//num_fold
remainder = len(train) % num_fold

folds = []
for i in range(num_fold):
    for j in range(sample_per_fold):
        folds.append(i+1)

for j in range(remainder):
    folds.append(j+1)

random.shuffle(folds)
#
#data = data.drop(["nocalldetection"],axis=1)

train['fold'] = folds
#data['target'] = targets
#data['durations'] = durations
#data["call_detection"] = call_binaries
#data = data[["filename","fold","target","primary_label","durations"]]

train.to_csv(input_path + 'stft_meta.csv')