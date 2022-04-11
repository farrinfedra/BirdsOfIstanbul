import pandas as pd
import random
import numpy as np
# read_csv function which is used to read the required CSV file
data = pd.read_csv('nocalldetection_for_shortaudio_fold0.csv')

print("Initial data shape: ")
print(data.shape)

# data = data.drop(["secondary_labels", "type", "latitude","longitude","scientific_name","common_name","author","date","license","url"], axis=1)
num_short_audio = data.shape[0]
maxLen = 10
#call_mat = pd.DataFrame(False,index=range(num_short_audio),columns=range(maxLen))
#call_mat = np.zeros((num_short_audio,maxLen),dtype=int)

call_binaries = []
for index, row in data.iterrows():
    probs = row['nocalldetection'].split()
    if len(probs) >= 10:
        probs = probs[0:10]
    probs_float = [float(x) for x in probs]
    arr = np.array(probs_float)
    arr_bool = (arr>0.5).astype(int)
    if len(arr_bool) < 10:
        arr_bool = np.append(arr_bool,np.zeros(maxLen - len(arr_bool),dtype=int)) 
    call_binaries.append(" ".join(str(x) for x in arr_bool))

data["call_detection"] = call_binaries
    
data.to_csv('test_call_mat.csv')



# data["fold"] = folds
# data["target"] = targets
# data = data[["filename","fold","target","primary_label"]]

