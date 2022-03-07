from tqdm import tqdm
from urllib import request, error
import json
import pandas as pd
import os
import requests

url = "https://xeno-canto.org/api/2/recordings?query=cnt:turkey%20loc:istanbul"
r = request.urlopen(url)

data = json.loads(r.read().decode('UTF-8'))
df = pd.DataFrame(data['recordings'])
df = df.drop(columns = ['lic', 'sono', 'rmk', 'playback-used', 'time', 'uploaded','date'])
df = df.loc[:, ['id','file-name','scientific-name', 'en', 'ssp','also', 'lat', 'lng', 'alt', 'cnt', 'loc', 'type', 'length','bird-seen','rec', 'file','url', 'q'  ]]

#rearrange downloaded metadata
df.to_csv('metadata.csv', index = False)

#prepare list to download the recordings
df2 = df.filter(['id','scientific-name', 'file'])
info = df2.to_dict('records')

for rec in tqdm(info, desc='downloading audio'):
    name = rec['scientific-name']
    base_path = './audio/'
    dir_name = name.replace(' ', '_')
    path = base_path + dir_name
    if not 'audio' in os.listdir('.'):
        os.mkdir('audio')
    if not dir_name in os.listdir('./audio'):
        os.mkdir(path)
    #download audio
    url = rec['file']
    re_url = requests.get(url, allow_redirects=True).url
    request.urlretrieve(url, path + '/' + rec['id'] + '.mp3')
