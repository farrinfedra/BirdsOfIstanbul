
# modified from:
# Author: Yuan Gong

# @File    : prep_custom.py


import numpy as np
import json
import os
import zipfile
import wget
import sys 
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


if os.path.exists('./data/custom_data/audio_16k') == False:
    #esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    #wget.download(esc50_url, out='./data/')
    #with zipfile.ZipFile('./data/ESC-50-master.zip', 'r') as zip_ref:
    #   zip_ref.extractall('./data/')
    #os.remove('./data/ESC-50-master.zip')
     
        # Lets say we manually upload code for now
        #  put data to ./data/custom_data/audio
        
    dataset_dir = '/datasets/birdclef-2021/train_short_audio'
    base_dir = './data/custom_data'
    os.mkdir('./data/custom_data/audio_16k/')
    for name in os.listdir(dataset_dir):
        curr_dir = os.path.join(dataset_dir, name)
    
    # convert the audio to 16kHz

        audio_list = get_immediate_files(curr_dir)
        for audio in audio_list:
            print('sox ' + curr_dir + '/' + audio + ' -r 16000 ' + base_dir + '/soundscape_audio_16k/' + audio)
            os.system('sox ' + curr_dir + '/' + audio + ' -r 16000 ' + base_dir + '/soundscape_audio_16k/' + audio)

#sys.exit()
            
label_set = np.loadtxt('./data/custom_labels_one_stage.csv', delimiter=',', dtype='str')

# fix bug: generate an empty directory to save json files
# if os.path.exists('./data/datafiles') == False:
#     os.mkdir('./data/datafiles')
    
datafiles_dir = './data/datafiles_one_stage_probs'

if os.path.exists(datafiles_dir) == False:
    os.mkdir(datafiles_dir)

nocall_label = 'a398'
    
    
for fold in [1]: # [1,2,3,4,5]
    base_path = "./data/custom_data/audio_16k/"
    #base_path = './data/fbank_tensors/'
    base_for_norm = "/kuacc/users/bbiner21/ast/egs/custom/data/custom_data/audio_16k/"
    meta = np.loadtxt('./data/custom_data/custom_meta_duration_probs.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):   # range(0, len(meta)) 200 was small dataset
        cur_labels = [x for x in meta[i][3].split("-")]
        cur_labels = ['a'+x.zfill(3) for x in cur_labels]
        
        # indices shifted by 1
        #cur_path = (meta[i][1]).split('.')[0] + '.pt'
        cur_path = meta[i][1]
        
        cur_fold = int(meta[i][2])
        curr_call_detection = (meta[i][5]).split() # 5 gives call_detection ones and zeros
        cur_latitude = float(meta[i][6])
        cur_longitude = float(meta[i][7])
        
        
        for cd_ind, cd_value in enumerate (curr_call_detection):
            if float(cd_value) >= 0.5 :
                cur_dict = {"wav": base_path + cur_path, "labels": "-".join(cur_labels),"segment":cd_ind, 'latitude':cur_latitude,'longitude':cur_longitude, 'prob': float(cd_value) }
                if cur_fold == fold:
                    eval_wav_list.append(cur_dict)
                else:
                    train_wav_list.append(cur_dict)
            else:
                cur_dict = {"wav": base_path + cur_path, "labels": nocall_label ,"segment":cd_ind, 'latitude':cur_latitude,'longitude':cur_longitude,'prob': (1.0-float(cd_value))  } #
                if cur_fold == fold:
                    eval_wav_list.append(cur_dict)
                else:
                    train_wav_list.append(cur_dict)
               
        
    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open(datafiles_dir + '/custom_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open(datafiles_dir + '/custom_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)
        
print('Finished custom Preparation')