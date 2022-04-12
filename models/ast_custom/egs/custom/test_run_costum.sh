#!/bin/bash
#SBATCH --partition ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=160G
#SBATCH --job-name="ast-custom"
#SBATCH --output=./log_%j.txt
#SBATCH --time=10:00:00

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#conda activate new_venv
export TORCH_HOME=../../pretrained_models

model=ast
dataset=custom
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96 # we use 700-128 fbanks so scaled accordingly but needs more thinking 
mixup=0
epoch=1
batch_size=36 #48 experiment with gpu memory 
fstride=10 
tstride=10

#python ./prep_custom.py


for((fold=1;fold<=1;fold++)); #1 fold for now
do
  echo 'now process fold'${fold}

  #exp_dir=${base_exp_dir}/fold${fold}
  te_data=./data/datafiles_train_soundscape/train_soundscape_nocall.json
  
  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/test_run.py --model ${model} --dataset ${dataset} \
  --data-test ${te_data} \
  --label-csv ./data/custom_labels_one_stage.csv --n_class 399\
  --batch-size $batch_size \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
done




#python ./get_esc_result.py --exp_path ${base_exp_dir}  # we may modify this later but not now