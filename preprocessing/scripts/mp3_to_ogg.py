import os 

path = '/datasets/xeno_canto/2022_validation/'

for bird in os.listdir(path):
    if bird == '2022_metadata.csv':
        continue
    for audio in os.listdir(path + bird):
        src = path + bird + '/' + audio
        new_audio = audio.replace('mp3', 'ogg')
        dest = path + bird + '/' + new_audio
        os.system("ffmpeg -i {src} {dest}".format(src = src, dest = dest))
        os.system("rm {src}".format(src = src))