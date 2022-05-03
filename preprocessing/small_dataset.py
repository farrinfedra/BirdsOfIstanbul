import os
import pandas as pd
import json

dest_audio = '/datasets/xeno_canto/491_dataset/'
dest_sm = '/datasets/xeno_canto/5-class/'

bird_list_path = '../../assets/combined_scientific_named_list.txt'
ebird_codes_path = '../../assets/eBird_Taxonomy_v2021.csv'


sm_dirs = os.listdir(dest_audio)
sm_dirs = sm_dirs[0:5]

if not os.path.exists(dest_sm):
    os.mkdir(dest_sm)

for dirs in sm_dirs:
    orig = dest_audio + dirs
    if dirs not in os.listdir(dest_sm):
        os.mkdir(dest_sm + dirs)
    dest = dest_sm
    if not len(os.listdir(dest_audio + dirs)) == 0:
        print("copying {orig}/* to {dest}".format(orig = orig, dest = dest))
        os.system("cp {orig}/* {dest}".format(orig = orig, dest = dest))