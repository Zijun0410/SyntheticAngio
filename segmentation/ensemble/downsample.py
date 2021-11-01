from imblearn.under_sampling import TomekLinks,ClusterCentroids,RepeatedEditedNearestNeighbours
from pathlib import Path 
import pandas as pd
import numpy as np
import pickle as pk
from collections import Counter
import os
from datetime import datetime
import argparse
from indexloader import IndexLoader
from baseclass import BaseDir
import re

class Downsample:
    """
    Apply Downsampling to pos and neg feature set
    Param:
       data_load_dir (Path() from pathlib): the directory where the date was loading from
       data_save_dir (Path() from pathlib): the directory where the date folder would be saved
       undersample_method (str): should be within 'TL'(TomekLinks), 'CC', 'RENN', 
                                          TL_CC'(+ ClusterCentroids)
                                          and 'TL_RENN'(+ RepeatedEditedNearestNeighbours)
       indexLoader (instance of the IndexLoader Class)
    """
    def __init__(self, data_load_dir, data_save_dir, undersample_method, indexLoader):
        
        self.data_load_dir = data_load_dir
        self.data_save_dir = data_save_dir / f'1_{undersample_method}' 
        self.undersample_method = undersample_method
        self.indexLoader = indexLoader
        self.make_directory_for_output()
        
    def make_directory_for_output(self):
        # Make directory for data output if they do not exist
        if not os.path.isdir(self.data_save_dir):
            os.mkdir(self.data_save_dir)
                
    def run_downsampling(self, idx_range=None):
        # Interate through features and apply downsampling
        if idx_range is None:
            idx_range = self.indexLoader.get_all_idx()

        for patient_index in idx_range:
            if not self.output_not_saved(patient_index):
                # Features are already saved
                continue 
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time) 

            print(f'Running the {patient_index}th....')
            X0 = np.genfromtxt(self.data_load_dir / f'{patient_index}-feature_neg.csv', delimiter=',')
            X1 = np.genfromtxt(self.data_load_dir / f'{patient_index}-feature_pos.csv', delimiter=',')
            X = np.concatenate((X1, X0), axis=0)
            y = np.array(len(X1)*[1] + len(X0)*[0])

            if 'TL' in undersample_method:
                print('-- TomekLinks')
                tk_undersample = TomekLinks(sampling_strategy='auto') # sample the majority class
                X, y = tk_undersample.fit_resample(X, y)

            if 'CC' in self.undersample_method:
                print('-- ClusterCentroids')
                number_list = re.findall(r'\d+', self.undersample_method)
                if len(number_list)==1:
                    cc_number = int(number_list[0])
                else:
                    cc_number = 4000
                print(f'---- The ClusterCentroids sampling number is {cc_number}')
                # try:
                #     cc_undersample = ClusterCentroids(sampling_strategy={1: cc_number, 0: cc_number}, random_state=42)
                #     X, y = cc_undersample.fit_resample(X, y)
                # except ValueError as e:
                cc_number_1 = min(cc_number,len(X1))
                cc_number_0 = min(cc_number,len(X0))
                cc_undersample = ClusterCentroids(sampling_strategy={1: cc_number_1, 0: cc_number_0}, random_state=42)
                X, y = cc_undersample.fit_resample(X, y)

                

            elif 'RENN' in self.undersample_method: 
                print('-- RepeatedEditedNearestNeighbours')
                renn_undersample = RepeatedEditedNearestNeighbours(max_iter=10000, sampling_strategy='all')
                X, y = renn_undersample.fit_resample(X, y)

            if self.output_not_saved(patient_index):
                self.save_to_sep_files(X, Counter(y), patient_index)

            
    def output_not_saved(self, patient_index):
        #  Returns:
        #    (bool) : if either the pos or neg features not saved
        pos_fname = self.data_save_dir / f'{patient_index}-feature_pos.csv' 
        neg_fname = self.data_save_dir / f'{patient_index}-feature_neg.csv'
        return not (os.path.isfile(pos_fname) and os.path.isfile(neg_fname))
    
    def save_to_sep_files(self, feature_array, labeling_counter, patient_index):
        #  Params:
        #    feature_array (numpy): concatnated feature matrix
        #    labeling_counter (instance of Counter, {1:num_pos, 0:num_neg})
        #    patient_index (int): count from 1
        pos_fname = self.data_save_dir / f'{patient_index}-feature_pos.csv' 
        neg_fname = self.data_save_dir / f'{patient_index}-feature_neg.csv'
        pos_features = feature_array[:labeling_counter[1],:]
        neg_features = feature_array[labeling_counter[1]:,:]
        np.savetxt(pos_fname, pos_features, delimiter=",")
        np.savetxt(neg_fname, neg_features, delimiter=",")   
        print(f'Output Saved...')

def main(undersample_method, range_param, target_folder='0_unsupervised'):

    dirHandle = BaseDir()
    base_data_dir = dirHandle.dir

    infor_name = 'patient.csv'
    indexLoader = IndexLoader(base_data_dir, infor_name)

    data_load_dir = base_data_dir / target_folder  
    downSample = Downsample(data_load_dir, base_data_dir, undersample_method, indexLoader)

    idx_range = list(range_param)
    downSample.run_downsampling(idx_range)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--start_index', default=1, type=int,
                      help='the start index of undersampling, a number from 1 to 130')
    args.add_argument('-n', '--index_number', default=20, type=int,
                      help='the number of indices for undersampling, a number from 1 to 130')    
    if not isinstance(args, tuple):
        # Use default value
        args = args.parse_args()
    start_index = args.start_index
    index_number = args.index_number

    # undersample_method = 'TL'
    undersample_method = 'TL_CC_4000_deepone'
    # undersample_method = 'TL_RENN'
    # undersample_method = 'CC'

    target_folder = '0_unsupervised_deep1'
    main(undersample_method, range(start_index,start_index+index_number),target_folder)