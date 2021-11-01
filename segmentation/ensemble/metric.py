
import pandas as pd
import numpy as np
from metric_util import calculate_metric

EPS = 1e-10

class MetricTracker:
    """
    A metric tracker of Trainer
    """
    def __init__(self, *keys):
        self.keywords = keys
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):

        for col in self._data.columns:
            self._data[col].values[:] = 0

        self._series = {}
        for key in self.keywords:
            self._series[key] = []

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]
        self._series[key].append(value)

    def avg(self, key):
        return self._data.average[key]

    def std(self):
        self._std = {}
        for key in self.keywords:
            all_data = np.array(self._series[key])  
            self._std[key] = np.std(all_data)
        return self._std

    def result(self):
        return dict(self._data.average)
    
    def get_data(self):
        return self._data

def iou(output, target):
    """Intersection over Union (IoU) metric = Jaccard Index"""
    output = np.array(output)
    target = np.array(target)

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    iou = (intersection + EPS) / (union + EPS)

    return iou

def auroc(output, target):
    _,_,_,_,_,_auroc = _cal_metric(output, target)
    return _auroc 

def precision(output, target):
    _precision,_,_,_,_,_ = _cal_metric(output, target)
    return _precision  
    
def accuracy(output, target):
    _,_accuracy,_,_,_,_ = _cal_metric(output, target)
    return _accuracy 

def sensitivity(output, target):
    _,_,_sensitivity,_,_,_ = _cal_metric(output, target)
    return _sensitivity    

def f1_score(output, target):
    _,_,_,_,_f1,_ = _cal_metric(output, target)
    return _f1

def specificity(output, target):
    _,_,_,_specificity,_,_ = _cal_metric(output, target)
    return _specificity

def _cal_metric(y_predict, y_true, method=5):
    """
    y_predict: predict value
    y_true: true label
    """
    precision,accuracy,sensitivity,specificity,F1,auroc = calculate_metric(y_true, y_predict,method)
    return precision,accuracy,sensitivity,specificity,F1,auroc

