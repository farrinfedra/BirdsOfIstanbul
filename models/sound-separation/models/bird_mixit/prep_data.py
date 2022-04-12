import os

path = '../../datasets/test_recordings/'

for audio in os.listdir(path):
    print(audio)
    src = path + audio
    new_audio = audio.replace('mp4', 'wav')
    dest = path + new_audio
    print('ffmpeg -i {src} -f wav {dest}'.format(src = src, dest = dest))
    os.system('ffmpeg -i {src} -f wav {dest}'.format(src = src, dest = dest))

    
def __