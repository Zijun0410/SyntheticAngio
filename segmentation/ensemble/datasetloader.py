import pickle
import os
import numpy as np
import pandas as pd


class DatasetLoader:
    """
    Load (and Save) the dataset for training, validation and testing 

    Param:
       sample_folder_name (string): The undersample output folder, e.g. '0_unsupervised', 
                                    '1_TL', '1_CC', '1_RENN', '1_TL_CC', '1_TL_RENN'
       base_data_dir (Path from pathlib): the directory where the csv was saved
       feature_number (int): Total number of features
    """
    def __init__(self, base_data_dir, undersample_name, feature_number, default_dimension=512):
        self.base_dir = base_data_dir
        self.undersample_name = undersample_name
        self.count_pd_dir = base_data_dir / 'scalers' / undersample_name / 'count.csv'

        self.data_load_dir = base_data_dir / undersample_name
        self.feature_number = feature_number
        self.dataset_name = ['nontest', 'test', 'train', 'validation']
        self.fold_name = ['fold_train', 'fold_test']
        self.default_dimension = default_dimension
        self.data_file_name_X = None

        self.init_flag = self.init_count_pd()
        
    def init_count_pd(self):
        # return 
        if os.path.isfile(self.count_pd_dir):
            self.count_pd = pd.read_csv(self.count_pd_dir)
            return False
        else:
            return True

    def get_init_flag(self):
        return self.init_flag

    def get_count_pd(self):
        return self.count_pd

    def get_base_dir(self):
        return self.base_dir
    
    def get_undersample_folder_name(self):
        return self.undersample_name
    
    def get_feature_number(self):
        return self.feature_number

    def init_target_id_and_file_name(self, indexLoader, dataset_case, fold_id=None):
        # Purpose:
        #    1. get the corresponding patient index of the target data
        #    2. get the file names of the target data 
        # Param:
        #    indexLoader (instance of IndexLoader Class): Load index for training, validationo
        #                 and testing set
        #    dataset_case (string): must be one of the element in self.dataset_name or self.fold_name
        #    fold_id (int): the fold number

        # flag_indicator = {True: '_full', False:''}
        if dataset_case in self.dataset_name:
            target_patient_id = indexLoader.get_idx(dataset_case)
            data_file_name_X  = self.data_load_dir / f'X_{dataset_case}.pkl' #{flag_indicator[full_flag]}
            data_file_name_y  = self.data_load_dir / f'y_{dataset_case}.pkl'
        elif dataset_case in self.fold_name and fold_id is not None:
            train_id, test_id = indexLoader.get_folds_idx(fold_id)
            if dataset_case=='fold_train':
                target_patient_id = train_id
            else:
                target_patient_id = test_id
            data_file_name_X  = self.data_load_dir / f'X_{dataset_case}_{fold_id}.pkl'
            data_file_name_y  = self.data_load_dir / f'y_{dataset_case}_{fold_id}.pkl'
        elif dataset_case == 'infer':
            target_patient_id = indexLoader.get_all_idx()
            data_file_name_X  = self.data_load_dir / f'X_{dataset_case}.pkl' #{flag_indicator[full_flag]}
            data_file_name_y  = self.data_load_dir / f'y_{dataset_case}.pkl'            
        else:
            print('DatasetLoader, check input!')
            return None    
        return target_patient_id, data_file_name_X, data_file_name_y

    def init_empty_datastruct(self, indexLoader, dataset_case, fold_id):
        # Purpose:
        #    1. init_target_id_and_file_name
        #    2. initate empty storage stucture for data loading
        # Param:
        #    indexLoader (instance of IndexLoader Class): Load index for training, validation
        #                 and testing set
        #    dataset_case (string): must be one of the element in self.dataset_name or self.fold_name or 'infer'
        #    fold_id (int): the fold number 

        self.target_patient_id, self.data_file_name_X, self.data_file_name_y = self.init_target_id_and_file_name(indexLoader, dataset_case, fold_id)

        target_df_idx = np.in1d(self.count_pd.index_one, self.target_patient_id)
        target_count_df = self.count_pd.loc[target_df_idx,:]
        # if full_flag:
        #     target_records_num = len(self.target_patient_id)*self.default_dimension*self.default_dimension
        # else:
        target_records_num = int(target_count_df.pos_count.sum()+target_count_df.neg_count.sum())

        self.X_target = np.zeros((target_records_num, self.feature_number))
        self.y_target = np.zeros(target_records_num)

        
    def load_undersampled_data(self, indexLoader, dataset_case, fold_id=None, save_flag=True):
        # Purpose:
        #    1. init_empty_datastruct
        #    2. fill in datastruct, return the raw X and label
        # Save format: under the data_load_dir, save as

        self.init_empty_datastruct(indexLoader, dataset_case, fold_id)

        if os.path.isfile(self.data_file_name_X):
            with open(self.data_file_name_X, 'rb') as handle:
                self.X_target = pickle.load(handle)
            with open(self.data_file_name_y, 'rb') as handle:
                self.y_target = pickle.load(handle)
        else:
            start_row = 0
            for patient_index in self.target_patient_id:
                print(f'DatasetLoader: Loading the {patient_index}th....')
                X0 = np.genfromtxt(self.data_load_dir / f'{patient_index}-feature_neg.csv', delimiter=',')
                X1 = np.genfromtxt(self.data_load_dir / f'{patient_index}-feature_pos.csv', delimiter=',')
                X = np.concatenate((X1, X0), axis=0)
                y = np.array(len(X1)*[1] + len(X0)*[0])
                self.X_target[start_row:start_row+len(X),:] = X
                self.y_target[start_row:start_row+len(X)] = y 
                start_row += len(X)
            if save_flag:
                with open(self.data_file_name_X, 'wb') as handle:
                    pickle.dump(self.X_target, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.data_file_name_y, 'wb') as handle:
                    pickle.dump(self.y_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.X_target, self.y_target

    def load_full_data(self, patient_index, deep1=False):
        # Purpose:
        #     Load all the feature for testing
        # Param:
        #     patient_index (int): start from 1
        #     deep1 (boolen): if deep1 is True, load features that generated from deep learning output
        #                     instead of deep feature maps of dimension 16

        full_data_load_dir = self.base_dir / 'features_21_16'
        label_load_dir = self.base_dir / 'label'
        X = np.genfromtxt(full_data_load_dir / f'{patient_index}-features.csv', delimiter=',')
        Y = np.genfromtxt(label_load_dir / f'{patient_index}-targets.csv', delimiter=',')
        return X,Y

    
    def load_unsupervised_data(self, patient_index):
        # Purpose:
        #     Load all the features after unsupervised undersampling for cross_validation
        # Param:
        #     patient_index(int): start from 1

        unsupervised_data_load_dir = self.base_dir / '0_unsupervised'
        X0 = np.genfromtxt(unsupervised_data_load_dir / f'{patient_index}-feature_neg.csv', delimiter=',')
        X1 = np.genfromtxt(unsupervised_data_load_dir / f'{patient_index}-feature_pos.csv', delimiter=',')        
        X = np.concatenate((X1, X0), axis=0)
        y = np.array(len(X1)*[1] + len(X0)*[0])

        return X, y 

    ## Full flag seems useless