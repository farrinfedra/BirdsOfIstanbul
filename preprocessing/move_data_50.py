import os
import pandas as pd


df = pd.read_csv('new_data.csv')
path = '/datasets/xeno_canto/491/'

src = df['path'].tolist()
dest = df['destination'].tolist()


for i,s in enumerate(src):
    
    os.system('cp {src} {d}'.format(src = s, d = dest[i]))
