#!/bin/bash
#python prep_data.py

python ../tools/process_wav.py \
--model_dir bird_mixit_model_checkpoints/output_sources8 \
--checkpoint bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900 \
--num_sources 8 \
--input ../../datasets/recordings/* \
--output ../../datasets/separated_recordings/audio.wav
#python3 ../tools/process_wav.py \
#--model_dir bird_mixit_model_checkpoints/output_sources4 \
#--checkpoint bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
#--num_sources 4 \
#--input ../../datasets/recordings/* \
#--output ../../datasets/separated_recordings/audio.wav