import os
import pandas as pd
import subprocess

base_path = '/datasets/xeno_canto/491_dataset/'


audio_dict = {}
birds = []
audio_names = []
lengths = []
paths = []
dest = []
base_path2 = '/datasets/xeno_canto/491/'
for bird in os.listdir(base_path):
    
    audios = os.listdir(base_path + bird)
    audio_names.extend(audios)
    temp = audios
    temp = [(base_path2 + a) for a in temp]
    dest.extend(temp)
    audios = [(base_path + bird + '/' + a) for a in audios]
    

    for audio in audios:
        command = 'ffprobe -i {audio} -show_entries format=duration -v quiet -of csv=p=0'.format(audio = audio)
        #print('processing {folder} and {audio}\n'.format(folder = bird, audio = audios))
        out = subprocess.Popen(command, 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT, shell = True)
        
        stdout,stderr = out.communicate()
        print(stdout)
        print(stderr)
        out = stdout.split()[0].decode("utf-8")
        birds.append(bird)
        paths.append(audio)
        lengths.append(out)
        

        
    
audio_dict['bird'] = birds
audio_dict['audio'] = audio_names
audio_dict['path'] = paths
audio_dict['length'] = lengths
audio_dict['destination'] = dest

df = pd.DataFrame(audio_dict)
df.to_csv('./durations.csv', index = False)