from pathlib import Path 
import pandas as pd
import numpy as np
import os

class IndexLoader:
    """
    Load training, testing, validation and fold Information from `patient.csv` under
    the `data` folder. 
    Param:
       base_dir (Path from pathlib): the directory where the csv was saved
       filename (str): the patient csv file name
          Need to contain columns: train_flag, test_flag, validation_flag, & fold_index
    """
    def __init__(self, base_dir, filename):
        self.inforFrame = pd.read_csv(base_dir / filename)
        self.numbers = np.array(range(1,len(self.inforFrame)+1))
        
    def get_all_idx(self):
        #  Returns
        #    (numpy array): range from 1 to number of patient in the csv
        return self.numbers
        
    def __len__(self):
        #  Returns
        #    (int): number of patients
        return len(self.numbers)

    def get_idx(self, case_name):
        #  Params:
        #    case_name (str): nontest, test, train, or validation
        #  Return:
        #    (numpy array): start from 1
        if case_name != 'nontest':
            return self.numbers[self.inforFrame[f'{case_name}_flag'].values.astype(int)==1]
        else:
            return self.numbers[self.inforFrame['test_flag'].values.astype(int)==0]

    def get_folds_idx(self, fold_id):
        #  Params:
        #    fold_id (int): 0, 1, .., K-1, where K is the number of folds
        #  Return:
        #    fold_train_idx_one, fold_valid_idx_one(numpy array): start from 1
        folds_idx = self.inforFrame['fold_index'].values.astype(int)
        fold_train_idx_one = self.numbers[np.logical_and(folds_idx!=fold_id, folds_idx>=0)]
        fold_valid_idx_one = self.numbers[folds_idx==fold_id]
        return fold_train_idx_one, fold_valid_idx_one

    def get_patient_idx(self, case_name):
        pass