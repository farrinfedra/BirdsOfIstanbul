import numpy as np
from scipy import stats
from sklearn import metrics
import torch
import csv
import sys

from typing import List

#import row_wise_micro_averaged_f1 as rw_micro


# from evaluations.kaggle_2020 import row_wise_micro_averaged_f1_score
"""
def row_wise_micro_averaged_f1_score(
        y_true: List[str],
        y_pred: List[str]
) -> float:
    """

    """
    n_rows = len(y_true)
    
    f1_score = 0.
    prec_score = 0.
    rec_score = 0.
    
    for true_row, predicted_row in zip(y_true, y_pred):
        
        f1, prec, rec = micro_f1_similarity(true_row, predicted_row)
        f1_score += f1 / n_rows
        prec_score += prec / n_rows
        rec_score += rec / n_rows
    return f1_score, prec_score, rec_score


def micro_f1_similarity(
        y_true: str,
        y_pred: str
) -> float:
    """

    """
    true_labels = y_true.split()
    pred_labels = y_pred.split()

    true_pos, false_pos, false_neg = 0, 0, 0

    for true_elem in true_labels:
        if true_elem in pred_labels:
            true_pos += 1
        else:
            false_neg += 1

    for pred_el in pred_labels:
        if pred_el not in true_labels:
            false_pos += 1
            
    if true_pos == 0:
        f1_similarity = 0.0
        precision_score = 0.0
        recall_score = 0.0
    else:
        f1_similarity = 2 * true_pos / (2 * true_pos + false_neg + false_pos)
        precision_score = true_pos /(true_pos+false_pos)
        recall_score = true_pos /(true_pos+false_neg)

    
        
    return f1_similarity, precision_score, recall_score


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

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

    samples_num, classes_num = target.shape
    print(samples_num)
    
    
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    # Class-wise statistics
#     for k in range(classes_num):
        
#         # Average precision
#         avg_precision = metrics.average_precision_score(
#             target[:, k], output[:, k], average=None)
        
# #         out_label = np.zeros(output[:,k].shape)
# #         for i, classification in enumerate (output[:,k]):
            
# #             if classification > 0.5:
# #                 out_label[i] = 1.0
                
#         arr = np.array(output[:,k])
#         out_label = (arr > 0.5).astype(float)

        
        
            
#         avg_f1 = metrics.f1_score(target[:, k], out_label, average='micro')


#         # AUC
# #         auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
#         auc = 0
#         # Precisions, recalls
#         (precisions, recalls, thresholds) = metrics.precision_recall_curve(
#             target[:, k], output[:, k])

#         # FPR, TPR
#         (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        
#         save_every_steps = 1000     # Sample statistics to reduce size
#         dict = {'precisions': precisions[0::save_every_steps],
#                 'recalls': recalls[0::save_every_steps],
#                 'AP': avg_precision,
#                 'fpr': fpr[0::save_every_steps],
#                 'fnr': 1. - tpr[0::save_every_steps],
#                 'auc': auc,
#                 # note acc is not class-wise, this is just to keep consistent with other metrics
#                 'acc': acc
#                 }
#         stats.append(dict)
    
#     label_csv = "/kuacc/users/bbiner21/ast/egs/custom/data/custom_labels.csv"
#     name_dict = make_name_dict(label_csv)
    
    preds = []
    true_labels = []
    for j in range(samples_num):
        arr = np.array(output[j,:])
        out_label = (arr > 0.5).astype(float)
        print(out_label)
        out = np.where(out_label)[0]
        arr_target = np.array(target[j,:])
        curr_target = np.where(arr_target)[0]
        
        out = [str(x) for x in out]
        out = " ".join(out)
        
        curr_target = [str(x) for x in curr_target]
        curr_target = " ".join(curr_target)
        
        
        #handling no call
#         if not out:
#             out = "398"
#         if not curr_target:
#             curr_target = "398"
#         preds.append(out)
#         true_labels.append(curr_target)
    
    f1_score, precision_score, recall_score = row_wise_micro_averaged_f1_score(true_labels,preds)
    
    
    stats.append(f1_score)
    stats.append(precision_score)
    stats.append(recall_score)
    
"""
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
    print('INSIDE STATS FUNCTION')
    classes_num = target.shape[-1]
    print("class num: {classs}".format(classs = classes_num))
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    
    return stats

