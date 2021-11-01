from pathlib import Path 
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import csv

import pdb

class ScaleLoader:
    """
    Generate or load the scale information for normalization or standarization.
    1. Scalers acorss folds : fold_{fold_idx}
    2. Scalers for training v.s. validation : validation 
    3. Scalers for nontesting v.s. testing : testing
    4. Scalers for all data v.s. inference : inference

    Param:
       base_dir (Path from pathlib): the directory where all the data were saved
       indexLoader (An instance of IndexLoader)
       feature_number (int): number of features 
       fold_number (int): number of folds in cross-validation
       undersample_name (str): the name of the undersample methods. Undersampling output are
                               saved under the folder
    """
    def __init__(self, base_dir, indexLoader, feature_number, fold_number, undersample_name, data_dir=None):
        self.indexLoader = indexLoader
        self.base_dir = base_dir
        self.undersample_name = undersample_name
        self.save_dir = base_dir / 'scalers' / undersample_name
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.patient_number = len(self.indexLoader)
        self.feature_number = feature_number
        self.count_dict = None
        self.standerizat_save = None
        self.normalizat_save = None  
        self.fold_number = fold_number
        
    def create_scaler(self, name_list=None):
        # Param:
        #    name_list (list of str): a list whose elements are within either ['validation','testing','inference'] 
        #                             or [f'fold_{fold_idx}' for fold_idx in range(fold_number)]
        if name_list is None:
            name_list = ['validation','testing','inference'] + [f'fold_{fold_idx}' for fold_idx in range(self.fold_number)]

        data_load_dir = self.base_dir / self.undersample_name

        #  ###################### INITIALIZATION ##########################
        # 1. Initialize Count the number of records
        count_dict = {}
        count_dict['index_one'] = self.indexLoader.get_all_idx()
        count_dict['pos_count'] = np.zeros(self.patient_number,)
        count_dict['neg_count'] =  np.zeros(self.patient_number,)
        # 2. Initialize Calculate the mean and variance 
        # 3. Initialize Record the max and min 
        standerizat_save = {}
        normalizat_save = {}
        for name in name_list:
            standerizat_save[name] = {'rolling_mean':np.zeros(self.feature_number,),
                                      'rolling_power_mean':np.zeros(self.feature_number,),
                                      'mean':np.zeros(self.feature_number,),
                                      'variance':np.zeros(self.feature_number,),
                                      'count': 0}
            normalizat_save[name] = {'max_value':np.zeros(self.feature_number,),'min_value':np.zeros(self.feature_number,)}
        # 4. Initialize belongs
        idx_belonging = {}
        idx_belonging['validation'] = self.indexLoader.get_idx('train')
        idx_belonging['testing']  = self.indexLoader.get_idx('nontest')
        idx_belonging['inference'] = self.indexLoader.get_all_idx()        
        for fold_idx in range(self.fold_number):
            idx_belonging[f'fold_{fold_idx}'], _ = self.indexLoader.get_folds_idx(fold_idx)

        #  ######################### INTERATION ###########################    
        # Start Interation
        for locat_idx, patient_idx in enumerate(self.indexLoader.get_all_idx()):
            print(f'ScaleLoader: Running the {patient_idx}th....')
            # Postive Records
            pos_features = data_load_dir / f'{patient_idx}-feature_pos.csv'
            pos_count = 0
            with open(pos_features) as file_handle:
                pos_csv_reader = csv.reader(file_handle)
                for row in pos_csv_reader:
                    pos_count += 1
                    for name in name_list:
                        if patient_idx in idx_belonging[name]:
                            test_output = np.asarray(row, dtype=np.float64)
                            # print(test_output.shape)
                            test_output_2 = standerizat_save[name]['rolling_mean']
                            # print(test_output_2.shape)
                            standerizat_save[name]['rolling_mean'] += np.asarray(row, dtype=np.float64)
                            standerizat_save[name]['rolling_power_mean'] += np.power(np.asarray(row, dtype=np.float64), 2)
                            standerizat_save[name]['count'] += 1
                            # 3. Record the max and min 
                            normalizat_save[name]['max_value'] = np.fmax(normalizat_save[name]['max_value'], np.asarray(row, dtype=np.float64))
                            normalizat_save[name]['min_value'] = np.fmin(normalizat_save[name]['min_value'], np.asarray(row, dtype=np.float64))
            
            # Negative Records
            neg_features = data_load_dir / f'{patient_idx}-feature_neg.csv'
            neg_count = 0
            with open(neg_features) as file_handle:
                neg_csv_reader = csv.reader(file_handle)
                for row in neg_csv_reader:
                    neg_count += 1
                    for name in name_list:
                        if patient_idx in idx_belonging[name]:
                            standerizat_save[name]['rolling_mean'] += np.asarray(row, dtype=np.float64)
                            standerizat_save[name]['rolling_power_mean'] += np.power(np.asarray(row, dtype=np.float64), 2)
                            standerizat_save[name]['count'] += 1
                            # 3. Record the max and min 
                            normalizat_save[name]['max_value'] = np.fmax(normalizat_save[name]['max_value'], np.asarray(row, dtype=np.float64))
                            normalizat_save[name]['min_value'] = np.fmin(normalizat_save[name]['min_value'], np.asarray(row, dtype=np.float64))
            # 1. Count the number of pos and neg records
            count_dict['pos_count'][locat_idx] = pos_count
            count_dict['neg_count'][locat_idx] = neg_count
            print(f'   There are {pos_count} positive samples and {neg_count} negtive samples')
        
        # 2. Calculate the mean and variance from sum and square sum (variance = E[x**2]-E[x]**2)
        for name in name_list:
            standerizat_save[name]['mean'] = np.divide(standerizat_save[name]['rolling_mean'], 
                                     standerizat_save[name]['count'])
            standerizat_save[name]['variance'] = np.divide(standerizat_save[name]['rolling_power_mean'], 
                                    standerizat_save[name]['count']) - np.power(standerizat_save[name]['mean'], 2)
        
        #  ######################### ASSIGN VALUE ###########################
        self.count_dict = count_dict
        self.standerizat_save = standerizat_save
        self.normalizat_save = normalizat_save
        
    def save_scaler(self, save_dir=None):
        # Save the scaler to file
        if save_dir is not None:
            self.save_dir = save_dir
        if self.standerizat_save is not None and self.normalizat_save is not None:
            
            count_save_file = self.save_dir / f'count.csv'
            count_dataframe = pd.DataFrame.from_dict(self.count_dict)
            count_dataframe.to_csv(count_save_file,index=False)
            
            name_list = list(self.standerizat_save.keys())
            for name in name_list:
                scale_save_file = self.save_dir  / f'scaler_{name}.csv'
                standard_dataframe = pd.DataFrame.from_dict(self.standerizat_save[name])
                norm_dataframe = pd.DataFrame.from_dict(self.normalizat_save[name])
                scaler_dataframe = pd.concat([standard_dataframe,norm_dataframe], axis=1)
                scaler_dataframe.to_csv(scale_save_file, index=False)
        else:
            print('No Data Found')
            
    def load_scaler(self, dataset_case, method, feature_range=(0,1)):
        # The major function in this Class that serves to load scaler data       
        # Param:
        #    dataset_case (string or int): a stirng within either ['validation','testing','inference'] 
        #                       or a integer that stands for one of the fold number
        #    method (stirng): stand, norm
        #    feature_range (tuple): optional, the range that the nomalization scaler map to
        
        # pdb.set__trace()

        valid_list = ['validation','testing','inference']
        if isinstance(dataset_case, int):
            dataset_case = f'fold_{dataset_case}'
        scale_save_file = self.save_dir  / f'scaler_{dataset_case}.csv'
            
        if not os.path.isfile(scale_save_file):
            self.create_scaler([dataset_case])
            self.save_scaler()

        scalers_pd = pd.read_csv(scale_save_file)

        if method=='stand':
            # TODo: some handling of the data
            scaler_instance = StandardScaler()
            scaler_instance.mean_ = scalers_pd['mean'].values
            scaler_instance.scale_ = np.sqrt(scalers_pd['variance'].values)

        elif method=='norm':
            scaler_instance = MinMaxScaler()
            data_max = scalers_pd['max_value'].values
            data_min = scalers_pd['min_value'].values
            data_range = data_max - data_min
            scaler_instance.scale_ = ((feature_range[1] - feature_range[0]) /
                           self._handle_zeros_in_scale(data_range))
            scaler_instance.min_ = feature_range[0] - data_min * scaler_instance.scale_
            scaler_instance.data_min_ = data_min
            scaler_instance.data_max_ = data_max
            scaler_instance.data_range_ = data_range
        return scaler_instance
    
    def scale_samples(self, data, scaler_instance):
        transformed_data = scaler_instance.transform(data)
        return transformed_data
    
    def _handle_zeros_in_scale(self, scale, copy=True):
        """Makes sure that whenever scale is zero, we handle it correctly.
        This happens in most scalers when we have constant features.
        """
        # if we are fitting on 1D arrays, scale might be a scalar
        if np.isscalar(scale):
            if scale == .0:
                scale = 1.
            return scale
        elif isinstance(scale, np.ndarray):
            if copy:
                # New array to avoid side-effects
                scale = scale.copy()
            scale[scale == 0.0] = 1.0
            return scale  