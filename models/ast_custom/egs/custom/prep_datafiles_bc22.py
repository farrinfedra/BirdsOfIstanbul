import numpy as np
import json
import os
import zipfile
import wget
import sys 
import pandas as pd
import csv

def make_name2mid_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['display_name']] = row['mid']
            line_count += 1
    return name_lookup

mid_dict = make_name2mid_dict('./bc-2022/bc_custom_labels_large.csv')


datafiles_dir = './data/datafiles_large_22'

if os.path.exists(datafiles_dir) == False:
    os.mkdir(datafiles_dir)    


#meta = np.loadtxt('./bc-2022/small_meta.csv', delimiter=',', dtype='str', skiprows=1) 

df = pd.read_csv('./bc-2022/large_meta.csv')
meta = df.values

segment_len = 5

base_path = '/kuacc/users/bbiner21/birdclef-2022/birdclef-2022-16khz/'

# label_list = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", "hawama", "hawcre", "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", "omao", "puaioh", "skylar", "warwhe1", "yefcan"]

label_list = (df['primary_label'] ).unique()


label_weights = {label: 0 for label in label_list}

for fold in [1,2,3,4,5]:
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):
        cur_label = str(meta[i][4])
        cur_file = meta[i][2]
        duration = meta[i][3]
        cur_fold = meta[i][5]
        cur_sec_labels = []
        for sec in eval(meta[i][1]): 
            cur_sec_labels.append(mid_dict[sec])

        segment_count = int(np.ceil(duration/segment_len))
        # /m/07rwj is just a dummy prefix

        for j in range(segment_count):
            cur_dict = {"wav": base_path + cur_file, "labels": 'a'+cur_label.zfill(3), 'segment': j,'sec_labels' : cur_sec_labels}
            
            if cur_fold == fold:
                eval_wav_list.append(cur_dict)
            else:
                train_wav_list.append(cur_dict)
                label_weights[meta[i][0]] += 1
    
    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open(datafiles_dir + '/train_data_' + str(fold) + '.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open(datafiles_dir + '/eval_data_' + str(fold) + '.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)
        
    with open(datafiles_dir + '/weights' + str(fold) + '.json', 'w') as f:
        json.dump({'weights': label_weights}, f, indent=1)
    
# with open('./data/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
#     json.dump({'data': eval_wav_list}, f, indent=1)

