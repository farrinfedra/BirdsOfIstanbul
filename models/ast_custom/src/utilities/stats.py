import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.
    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []
    
    out_binary = torch.where(output <= 0.5, torch.tensor(0),  torch.tensor(1))

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # New way of calculatin metrics:

        
    precision_micro = metrics.precision_score(target, out_binary, average='micro')
    precision_macro = metrics.precision_score(target, out_binary, average='macro')
    precision_classes = metrics.precision_score(target, out_binary, average=None)

    
    
    
    recall_micro = metrics.recall_score(target, out_binary, average='micro')
    recall_macro = metrics.recall_score(target, out_binary, average='macro')
    recall_classes = metrics.recall_score(target, out_binary, average=None)
        
    f1_micro = metrics.f1_score(target, out_binary, average='micro')
    f1_macro = metrics.f1_score(target, out_binary, average='macro')
    f1_classes = metrics.f1_score(target, out_binary, average=None)
    
    dict = {
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_classes' : precision_classes,
        'recall_micro' : recall_micro,
        'recall_macro' : recall_macro,
        'recall_classes' : recall_classes,
        'f1_micro' : f1_micro, 
        'f1_macro' : f1_macro,
        'f1_classes' : f1_classes,
        'acc' : acc
        }
    
    stats.append(dict)
    

    return stats