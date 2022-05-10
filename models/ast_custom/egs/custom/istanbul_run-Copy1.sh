#!/bin/bash
#SBATCH --partition ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_t4:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=190G
#SBATCH --job-name="istangull"
#SBATCH --output=./log_%j.txt
#SBATCH --time=24:00:00

module load anaconda/3.21.05
module load gcc/9.3.0
set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#conda activate new_venv
export TORCH_HOME=../../pretrained_models

model=ast
dataset=custom
nocall=True
timetest = True
imagenetpretrain=True
audiosetpretrain=True
bal=False
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96 # 96 -> 5 seconds 135 -> 7 seconds 
mixup=0
epoch=6
batch_size=48 #48 experiment with gpu memory 
fstride=10 
tstride=10
base_exp_dir=./exp/train-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}-$epoch-istangull_50class_solid

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

for((fold=1;fold<=1;fold++)); #change this to 5 folds
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/custom_train_data_${fold}.json
  te_data=./data/datafiles/custom_eval_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/custom_labels.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
done



#python ./get_esc_result.py --exp_path ${base_exp_dir}  # we may modify this later but not now