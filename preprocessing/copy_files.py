"""
Convert bird list to scientific name ~done
Get folders from /datasets/xeno_canto/audio to /datasets/xeno_canto/491_dataset
Change folder names to bird code

"""

import os
import pandas as pd
import json

xc_audio = '/datasets/xeno_canto/audio/'
xc_metadata = '/datasets/xeno_canto/metadata/'
dest_audio = '/datasets/xeno_canto/491_dataset/'

bird_list_path = '../assets/combined_scientific_named_list.txt'
ebird_codes_path = '../assets/eBird_Taxonomy_v2021.csv'

all_audio = os.listdir(xc_audio)
all_meta = os.listdir(xc_metadata)

bird_list = []
with open(bird_list_path, 'r') as blist:
    temp = blist.readlines()
    bird_list = [f.replace('\n', '') for f in temp]
    bird_list.sort()

df = pd.read_csv(ebird_codes_path)

#function get name
def get_bird_code(bird):
    bird = bird.replace('_', ' ').capitalize()
    code = df[df.SCI_NAME == bird].SPECIES_CODE.values[0]
    return code
    

def copy_audio():
    for bird in bird_list:
        if bird in all_audio:
            code = get_bird_code(bird)
            base = '/datasets/xeno_canto/'
        
            if not '491_dataset' in os.listdir(base):
                os.mkdir(base + '491_dataset')
        
            if not bird in os.listdir(dest_audio):
                os.mkdir(dest_audio + code)
        
            src = xc_audio + bird
            dst = dest_audio + code
            #copy file form audio to 491_dataset
            print("Currently in folder {name}\n".format(name = bird))
            
            #os.system('cp {src}/* {dest}'.format(src = src, dest = dst))
            for audio in os.listdir(xc_audio + bird):
                audio_path = xc_audio + bird + '/' + audio
                new_audio_path = dest_audio + code + '/' + audio.replace('mp3', 'wav')
                print("Converting {audio} to {new_audio}\n".format(audio = audio_path, 
                                                                new_audio = new_audio_path))
                os.system('ffmpeg -i {audio} -ar 16000 {new_audio}'.format(audio = audio_path, 
                                                                          new_audio = new_audio_path))
            print("Finished processing file {name}\n".format(name = bird))
            print("-------------------------------------------------------------------------------------")

def copy_metadata():
    output_list = []
    for bird in bird_list:
        if bird in all_meta:
            for json_file in os.listdir(xc_metadata + bird):
                
                with open(xc_metadata + bird + '/' + json_file, "rb") as infile:
                    load_file = json.load(infile)
                    rec_dict = load_file['recordings']
            
                for elem in rec_dict:
                    elem['primary_label'] = get_bird_code(bird)
                    elem['secondary_label'] = elem['also']
                    elem['latitude']= elem['lat']
                    elem['longitude']= elem['lng']
                    elem['scientific_name'] = elem['gen'] + ' ' + elem['sp']
                    elem['common_name'] = elem['en']
                    idd = elem['id'] #not include
                    file = 'XC{ids}.ogg'.format(ids = idd) #not include
                    elem['filename'] = bird + '/' + file
                    del elem['gen']
                    del elem['sp']
                    del elem['ssp']
                    del elem['cnt']
                    del elem['loc']
                    del elem['rmk']
                    del elem['also']
                    del elem['en']
                    del elem['lat']
                    del elem['lng']
                    output_list.append(elem)
        return output_list

def process_metadata(output_list, path):
    df = pd.DataFrame.from_dict(output_list)
    df.drop(columns = ['id', 'rec', 'file', 'file-name', 'sono',
                   'lic', 'uploaded', 'bird-seen', 'playback-used', 'alt', 'length'])
    df = df[['primary_label', 'secondary_label', 'type', 'latitude', 'longitude', 'scientific_name', 'common_name',
            'time', 'url', 'filename', 'q']]
    
    df.to_csv(path, index = False)
    

if __name__ == '__main__':
    copy_audio()
    #output = copy_metadata()
    #process_metadata(output, '../../dataset/metadata/metadata.csv')
    