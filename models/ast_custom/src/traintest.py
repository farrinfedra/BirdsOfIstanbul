# modified from:
# Author: Yuan Gong

# @File    : traintest.py

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




def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP, best_f1 = 0, 0, -np.inf, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)


    path = '/kuacc/users/fsofian19/COMP491_model/models/ast_custom/pretrained_models/bc_21_5s_best.pth' #laod previous pretrained model
    sd = torch.load(path, map_location=device)


    audio_model.load_state_dict(sd,strict=False)
    print('state dict to model completed => pretrained weights loaded')
    print(audio_model.named_parameters())


    #for name, param in audio_model.named_parameters(): #delete
    #    if 'module.v' in name:   # freezing the layers before classification 
    #        param.requires_grad = False
    #    else:
    #        print('not freezing:')
    #        print(name)
    #        param.requires_grad = True
    #sys.exit()
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    if args.dataset == 'audioset':
        if len(train_loader.dataset) > 2e5:
            print('scheduler for full audioset is used')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)
        else:
            print('scheduler for balanced audioset is used')
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
        main_metrics = 'mAP'
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = True
    elif args.dataset == 'esc50':
        print('scheduler for esc-50 is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        main_metrics = 'acc'
        loss_fn = nn.CrossEntropyLoss()
        warmup = False
    elif args.dataset == 'speechcommands':
        print('scheduler for speech commands is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
        main_metrics = 'acc'
        loss_fn = nn.BCEWithLogitsLoss()
        warmup = False
    elif args.dataset == 'custom':
        print('scheduler for custom dataset is used')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85) # we may modify this later
        main_metrics = 'mAP'
        loss_fn = nn.BCEWithLogitsLoss(reduction = 'none')#reduction = 'none'
        warmup = False
    else:
        raise ValueError('unknown dataset, dataset should be in [audioset, speechcommands, esc50,custom]')
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    args.loss_fn = loss_fn

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        end_time = time.time()
        for i, (audio_input, labels) in enumerate(train_loader):
            
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            #label_weights = label_weights.to(device, non_blocking=True)
        
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                #print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                audio_output = audio_model(audio_input)
                
                #print("audio_output is {out}\n".format(out = audio_output))
                
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                    #print("USING CROSS ENTROPY LOSS")
                else:
                    loss = loss_fn(audio_output, labels)
                    #print('LOSS is {loss}'.format(loss = loss))
                    
            
            loss = loss.mean()
            #loss = (loss * label_weights).mean() # secondary labels are ignored
            #print('LOSS after label_weights multiplication is {loss}'.format(loss = loss))
            print('LOSS is {loss}'.format(loss = loss))
            
            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                
                
                wandb.log({"Train loss":loss_meter.avg})
                
                #gc.collect()

                #subprocess.call("nvidia-smi",shell=True)

                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

                
            global_step += 1  
            ##here sth
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)
        """
        f1 = stats[0]
        avg_precision = stats[1]
        avg_recall = stats[2]

        print("f1_score: {:.6f}".format(f1))
        print("avg_precision_score: {:.6f}".format(avg_precision))
        print("avg_recall_score: {:.6f}".format(avg_recall))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        
        wandb.log({'validation f1_score':f1, "validation avg_precision_score":avg_precision, "validation avg_recall_score":avg_recall})
        """
        stat_dict = stats[0]

        
        
        
        precision_micro = stat_dict['precision_micro']
        precision_macro = stat_dict['precision_macro']
        precision_classes = stat_dict.pop('precision_classes')
        
        recall_micro = stat_dict['recall_micro']
        recall_macro = stat_dict['recall_macro']
        recall_classes = stat_dict.pop('recall_classes')
        
        f1_micro = stat_dict['f1_micro']
        f1_macro = stat_dict['f1_macro']
        f1_classes = stat_dict.pop('f1_classes')
        
        
        print("Precision Micro: {:.6f}".format(precision_micro))
        print("Precision Macro: {:.6f}".format(precision_macro))
        print("Recall Micro: {:.6f}".format(recall_micro))
        print("Recall Macro: {:.6f}".format(recall_macro))
        print("F1 Micro: {:.6f}".format(f1_micro))
        print("F1 Micro: {:.6f}".format(f1_macro))
        
        print('classes- precision - recall - f1:')
        
        print(precision_classes)
        print(recall_classes)
        print(f1_classes)
        
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        
        print('validation finished')

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_epoch = epoch
            
        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
            
        """
        
        if f1 > best_f1:
            best_f1 = f1
            if main_metrics == 'mAP':
                best_epoch = epoch


        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))


        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
"""
        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

#         with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
#             pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1
#         subprocess.call("nvidia-smi",shell=True)
        
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()


def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    
    print("inside validate")
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            #audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            #A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            #label_weights = label_weights.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            
            #loss = (loss * label_weights).mean()
            loss = loss.mean()
            print('LOSS in validation is {loss}'.format(loss = loss))
            A_loss.append(loss.to('cpu').detach())
            
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()
            A_predictions.append(predictions)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
        
        audio_output = torch.cat(A_predictions)
        #print(audio_output)
        target = torch.cat(A_targets)
        #loss = np.mean(A_loss) #commented this out since we are using mean above
        
        stats = calculate_stats(audio_output, target)
        # save the prediction here
        print("\n----------------------------------------STATS-----------------------------------\n")
        print(stats)
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

