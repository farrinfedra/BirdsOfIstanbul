import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import subprocess
import gc
import wandb
import argparse
import ast

import pandas as pd


import models
import dataloader_custom
import wandb

import dataloader_custom



def test(audio_model, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/best_audio_model.pth', map_location=device) # enter pth path here
    
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.load_state_dict(sd)
    
    print("load state dict completed")
    batch_time = AverageMeter()

    # switch to evaluate mode
    audio_model.eval()


    df_call_detection = pd.DataFrame()
    filenames = []
    segments = []
    birdcall_probs = []
    birdcall_binary = []
    
    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    
    softmax = nn.Softmax(dim=1) # 0 col 1 row

    with torch.no_grad():
        for i, (audio_input, labels, sec_labels ) in enumerate(test_loader):
            audio_input = audio_input.to(device)
            # compute output
            audio_output = audio_model(audio_input)
            
#             filenames += filename #  (batch size * 2) is expected as size
#             segments += segment.tolist()
            
            #audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)
            #batch_birdcall_probs = softmax(audio_output)[:,1]
            #birdcall_probs += batch_birdcall_probs.tolist()
            
#             birdcall = torch.argmax(audio_output, dim=1) 
#             birdcall_binary += birdcall.tolist()



            # compute the loss
#             labels = labels.to(device)
#             if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
#                 loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
#             else:
#                 loss = args.loss_fn(audio_output, labels)
#             A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()
        
#         df_call_detection['filenames'] = filenames
#         df_call_detection['segments'] = segments
#         df_call_detection['birdcall'] = birdcall_binary
        
#         df_call_detection.to_csv('/kuacc/users/bbiner21/ast/egs/custom/nocall_detection_soundscapes_5smodel.csv')
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        #loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)
        
        # save the prediction here
#         exp_dir = args.exp_dir
#         if os.path.exists(exp_dir+'/predictions') == False:
#             os.mkdir(exp_dir+'/predictions')
#             np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
#         np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')
    loss = 0
    return stats, loss


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--data-test", type=str, default='', help="test data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--dataset", type=str, default="custom", help="the dataset used", choices=["audioset", "esc50", "speechcommands","custom"])
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument("--n_class", type=int, default=399, help="number of classes")
parser.add_argument('-b', '--batch-size', default=36, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')


args = parser.parse_args( )
print("inside test run")
args.loss_fn = nn.BCEWithLogitsLoss()
#args.exp_dir = '/kuacc/users/bbiner21/ast/egs/custom/exp/test-custom-f10-t10-impTrue-aspTrue-b36-lr1e-5-25/fold1'
args.exp_dir = '/kuacc/users/bbiner21/ast/egs/stft_transformer/exp/test-custom-f10-t10-impTrue-aspTrue-b8-lr1e-5-20-stft/fold1/models'

#args.data_test = '/kuacc/users/bbiner21/ast/egs/custom/data/datafiles_train_soundscape/train_soundscape_.json'

#args.label_csv = '/kuacc/users/bbiner21/ast/egs/custom/data/custom_labels.csv'


norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526],'custom':[-6.0953665, 4.78203]}
#-3.8617978, 2.849493
#-6.0953665, 4.78203

target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128,'custom':576} # 700 normally

audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')

test_audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False} # 500 -> 700

test_loader = torch.utils.data.DataLoader(
    dataloader_custom.AudiosetDataset(args.data_test, label_csv=args.label_csv, audio_conf=test_audio_conf),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

stats, loss = test(audio_model, test_loader, args)


f1 = stats[0]
avg_precision = stats[1]
avg_recall = stats[2]

print("f1_score: {:.6f}".format(f1))
print("avg_precision_score: {:.6f}".format(avg_precision))
print("avg_recall_score: {:.6f}".format(avg_recall))



