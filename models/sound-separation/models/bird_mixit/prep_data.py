import os
import wave

def convert_audio(path, name):
    rate = 25050
    old_path = path + name
    dest = path + name.replace('.wav', '_c.wav')
    
    os.system('ffmpeg -i {src} -ar {rate} {dest}'.format(src = old_path, rate = rate,
                                                        dest = dest))
    os.system('rm {old}'.format(old = old_path))

    
if __name__ == '__main__':
    path = '../../datasets/test_recordings/'
    files = os.listdir(path)
    for audio in files:
        if '.wav' in audio:
            with wave.open((path + audio), "rb") as fs:
                srate = fs.getframerate()
            if not srate == 25050:
                convert_audio(path, audio)
            else:
                break
    print("All data converted to 22050 Hz")
    