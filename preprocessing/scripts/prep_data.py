import os
from tqdm import tqdm

fun = input("Enter wav for wav conversion and resample for resampling the audio\n")

inputs = input("Enter input directory and output directory in the following format: /path/to/input /path/to/output\n")
if not fun == 'wav' :
    sample_input = input('Enter sample rate in hz\n')

input_dir, output_dir = inputs.split()
species = os.listdir(input_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def convert_wav():
    for bird in species: #list of species folder
        if os.path.isfile(input_dir + '/' +bird):
            continue
        audios = os.listdir(input_dir + '/' + bird)
        if "in_progress.txt" in audios:
            audios.remove("in_progress.txt")
    
        if not bird in os.listdir(output_dir):
            os.mkdir(output_dir + '/' + bird)
        folders = tqdm(audios)
        for audio in folders:
            folders.set_description("Processing audios in %s" % bird)
            name = audio.split('.')[0]
            #print('lame --decode --quiet ' + input_dir + '/' +  bird + '/' + audio + ' '+ output_dir + '/' + bird + '/' + name + '.wav')
            os.system('lame --decode --quiet ' + input_dir + '/' +  bird +  '/' + audio + ' '+ output_dir + '/' +  bird + '/' + name + '.wav')


def resample(sample_input):
    for bird in species: #list of species folder
        if os.path.isfile(input_dir + '/' +bird):
            continue
        audios = os.listdir(input_dir + '/' + bird)
        if "in_progress.txt" in audios:
            audios.remove("in_progress.txt")
            continue
        if "DS_Store" in audios:
            audios.remove("DS_Store")
            continue
        folders = tqdm(audios)
        for audio in folders:
            folders.set_description("Processing audios in %s" % bird)
            os.system('sox -r ' + sample_input + ' ' + input_dir + '/' +  bird +  '/' + audio + ' ' + output_dir + '/' + bird + '/' + audio)

if fun == 'wav':
    convert_wav()
if fun == 'resample':
    resample(sample_input)
