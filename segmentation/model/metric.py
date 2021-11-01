import torch

import numpy as np

from .metric_util import calculate_metric, otsu

# From https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/metrics.py
EPS = 1e-10


def iou(output, target, method=5):
    """Intersection over Union (IoU) metric = Jaccard Index"""
    with torch.no_grad():
        try:
            output = np.array(flatten(output.cpu()))
            target = np.array(flatten(target.cpu()))
        except RuntimeError:
            # For CPU running 
            y_true = flatten(target.detach().numpy())
            y_predict = flatten(output.detach().numpy())

        if method == 5:
            threshold = otsu(output)
        else:
            threshold = 0.5

        output_ = output > threshold
        target_ = target > threshold
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()

        iou = (intersection + EPS) / (union + EPS)

        return iou

def auroc(output, target, method = 5):
    _,_,_,_,_,_auroc = _cal_metric(output, target, method)
    return _auroc 

def precision(output, target, method = 5):
    _precision,_,_,_,_,_ = _cal_metric(output, target, method)
    return _precision  
    
def accuracy(output, target, method = 5):
    _,_accuracy,_,_,_,_ = _cal_metric(output, target, method)
    return _accuracy 

def sensitivity(output, target, method = 5):
    _,_,_sensitivity,_,_,_ = _cal_metric(output, target, method)
    return _sensitivity    

def f1_score(output, target, method = 5):
    _,_,_,_,_f1,_ = _cal_metric(output, target, method)
    return _f1

def specificity(output, target, method = 5):
    _,_,_,_specificity,_,_ = _cal_metric(output, target, method)
    return _specificity

def _cal_metric(output, target, method):
    """
    output: predict value
    target: true label
    """
    with torch.no_grad():
        try:
            y_true = np.array(flatten(target.cpu()))
            y_predict = np.array(flatten(output.cpu()))
        except RuntimeError:
            # For CPU running
            y_true = flatten(target.detach().numpy())
            y_predict = flatten(output.detach().numpy())

        precision,accuracy,sensitivity,specificity,F1,auroc = calculate_metric(y_true, y_predict, method)
        return precision,accuracy,sensitivity,specificity,F1,auroc

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t