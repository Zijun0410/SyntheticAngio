import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
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
        if not np.isnan(value):
            if self.writer is not None:
                self.writer.add_scalar(key, value)
            self._data.total[key] += value * n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key]
            self._series[key].append(value)

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

    def std(self):
        self._std = {}
        for key in self.keywords:
            all_data = np.array(self._series[key])  
            self._std[key] = np.nanstd(all_data)
        return self._std

    def get_data(self):
        return self._data
