from indexloader import IndexLoader
from scaleloader import ScaleLoader
from baseclass import BaseDir
from datasetloader import DatasetLoader

import pickle
import platform
from pathlib import Path 
import pandas as pd
import numpy as np
import os
import itertools
from deepforest import CascadeForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from skimage.io import imsave
from datetime import datetime
import argparse

from joblib import Parallel, delayed

import pdb

from metric import MetricTracker
from metric import precision, accuracy, sensitivity, specificity, f1_score, auroc, iou

class Trainer:
    """
    Trainer Class: perform cross-validation for hyper-param selection and final tesing process
    """
    def __init__(self, indexLoader, datasetLoader, split_dict):
        # Construct the scaleLoader as a class attribute
        self.base_data_dir = datasetLoader.get_base_dir()
        self.undersample_name = datasetLoader.get_undersample_folder_name()
        self.folds_number = split_dict['TRAIN'] + split_dict['VALID']
        self.feature_number = datasetLoader.get_feature_number()
        self.scaleLoader = ScaleLoader(self.base_data_dir, indexLoader, feature_number=self.feature_number, 
                          fold_number=self.folds_number, undersample_name=self.undersample_name)

        # Add the indexLoader and datasetLoader as class attributes
        self.indexLoader = indexLoader
        self.datasetLoader = datasetLoader

        # Initiate Scales if not exist
        if self.datasetLoader.get_init_flag():
            self.scaleLoader.create_scaler()
            self.scaleLoader.save_scaler()
        
            # Initiate the scaler in datsetLoader
            _ = self.datasetLoader.init_count_pd()

        # Metric Infor
        metric_func = [precision, accuracy, sensitivity, specificity, f1_score, auroc, iou]
        metric_name = ['Precision', 'Accuracy','Sensitivity','Specificity','F1Score','AUROC','IOU']
        self.metric_dict = dict(zip([m.__name__ for m in metric_func],metric_name))

    def load_scaler(self, dataset_indicator, method, feature_range=(0,1)):
        # Purpose:
        #    Return a scaler instance to scale the training/testing data
        # Param:
        #    dataset_indicator (string or int): a stirng within either ['validation','testing','inference'] 
        #                       or a integer that stands for one of the fold number
        #    method (stirng): stand, norm
        #    feature_range (tuple): optional, the range that the nomalization scaler map to

        # return a scaler instance 
        return self.scaleLoader.load_scaler(dataset_indicator, method)

    def train(self, X, y, modelHandle, model_name, dataset_indicator):
        # Purpose:
        #    1. Train the model
        #    2. Save model 
        #    3. return the trained model handle
        # Params:
        #    X, y: Training data and labels
        #    modelHandle: a model handle (with some hypar-param)
        #    model_name (string): a name for saving 
        #    dataset_indicator (string or int): a stirng within either ['validation','testing','inference'] 
        #                       or a integer that stands for one of the fold number

        modelHandle.fit(X, y)
        self.save_model(modelHandle, model_name, dataset_indicator)
        return modelHandle


    def return_save_dir(self, dataset_indicator):
        # Purpose:
        #    1. Make dir to save the model (trained model) and the predict results
        #    2. Return these dirs
        # Params:
        #    dataset_indicator (string or int): a stirng within either ['validation','testing','inference'] 
        #                       or a integer that stands for one of the fold number

        model_dir = self.base_data_dir / 'model' / self.undersample_name
        result_dir = self.base_data_dir / 'result' / self.undersample_name

        if isinstance(dataset_indicator,int):
            model_dir = model_dir / f'{dataset_indicator}'
            result_dir = result_dir / f'{dataset_indicator}'

        ## TODO: check if it works as expected
        if not os.path.isdir(model_dir):
            model_dir.mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(result_dir):
            result_dir.mkdir(parents=True, exist_ok=True)

        return model_dir, result_dir

    def save_model(self, modelHandle, model_name, dataset_indicator):
        # Purpose:
        #    Save trained model 

        model_dir, _ = self.return_save_dir(dataset_indicator)       
        pickle.dump(modelHandle, open( model_dir / f'{model_name}.sav', 'wb'))
        print('Trainer: Model Saved')
        
    def load_model(self, model_name, dataset_indicator):
        # Purpose:
        #    1. Load model if it exist
        #    2. Return None if it does not

        model_dir, _ = self.return_save_dir(dataset_indicator) 
        model_save_path = model_dir / f'{model_name}.sav'
        if os.path.isfile(model_save_path):
            try:
                modelHandle = pickle.load(open(model_save_path, 'rb'))
            except EOFError as e:
                # Just in case sometime it's empty
                return None
            return modelHandle
        return None

    ################### Cross Validation #################

    def init_model_result_saver(self, model_params_opts, folds_number, metric_ftns):
        # Purpose:
        #    Init a pandas data frame (the data sturcture for cv results) with a regard to a specific fold number
        # Param:
        #    model_params_opts (dict):
        #                     {'learning_rate': [0.01, 0.05, 0.1],
        #                      'max_depth':[3, 5, 10, 20]}  
        #    folds_number (int): the fold number that we are intereted in. [start from 0]
        #    metric_ftns (list of functions): the set of metric that we are interested in 

        self.param_names = list(model_params_opts.keys())
        self.param_combination = list(itertools.product(*list(model_params_opts.values())))
        # param_combination = [(0.01, 3), (0.01, 5), (0.01, 10), ...]
        model_params_opts['folds'] = [folds_number]
        
        metric_df = pd.DataFrame(list(itertools.product(*list(model_params_opts.values()))), 
            columns=list(model_params_opts.keys()))
        metrics = [self.metric_dict[met.__name__] for met in metric_ftns]
        metric_df = metric_df.reindex(columns=metric_df.columns.tolist() + metrics)
        return metric_df

    def get_param_combination_count(self):
        # Purpose:
        #   helper function 
        return len(self.param_combination)

    def get_metric_dict(self):
        # Purpose:
        #   helper function 
        return self.metric_dict

    def train_across_iParm(self, iPram, fold_number, model_class, model_name, 
        metric_ftns, save_image=False):
        # Purpose:
        #    A parallelable function for Cross Validation 
        # Param:
        #    iPram (int): the ith parameter combination of the jth fold_number
        #    folds_number (int): the fold number that we are intereted in. [start from 0]
        #    model_class: a class instance of the target model
        #    model_name (string): the name of the target model
        #    metric_ftns (list of functions): the set of metric that we are interested in, should
        #             be the same with those from the function init_model_result_saver

        
        ## 1. Check if there are needs to run the loop
        _, result_dir = self.return_save_dir(fold_number)
        save_file_name = result_dir / f'cv{fold_number}_results{iPram+1}.csv'
        if os.path.isfile(save_file_name):
            print(f"The {iPram}th param is already done.")
            return 0

        ## 2. Set model param (for 3. and 7.)
        crrent_param = dict(zip(self.param_names, self.param_combination[iPram]))

        ## 3. Print out all the information 
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"   Running the {iPram+1}th param."," Current Time = ", current_time)
        print("           Model Infor: ") 
        for param_name in self.param_names:
            print(f"           {param_name}:{crrent_param[param_name]}")            
        
        ## 4. Init MetricTracker
        validation_metrics = MetricTracker(*[m.__name__ for m in metric_ftns])
        validation_metrics.reset()

        ## 5. Load (in fold) Training Data
        X_train, y_train = self.datasetLoader.load_undersampled_data(self.indexLoader, 'fold_train',fold_number)

        ## 6. Scale (in fold) Training Data
        scaler_instance = self.load_scaler(fold_number,'stand')
        X_train = self.scaleLoader.scale_samples(X_train, scaler_instance)

        ## 7. Init model handle
        crrent_param['verbose'] = 1
        model_handle = model_class(**crrent_param)
        
        ## 8. Train model (or load if it exists)
        model_name_save = f"{model_name}_{iPram+1}"
        model_trained = self.load_model(model_name_save, fold_number)
        if model_trained is None:
            kwargs = {"modelHandle":model_handle,"model_name":model_name_save, "dataset_indicator":fold_number}
            model_handle = self.train(X_train, y_train, **kwargs)
            self.save_model(model_handle, model_name_save, fold_number) 
        else:
            model_handle = model_trained   

        ## 9. Validation 
        # load the patient index for validation  
        target_patient_id, _, _ = self.datasetLoader.init_target_id_and_file_name(self.indexLoader, 'fold_test', fold_number)
        for patient_id in target_patient_id:
            
            # load unsupvervised output for validtaion 
            X_valid, y_valid_label = self.datasetLoader.load_unsupervised_data(patient_id)
           
            # scale sample
            X_valid = self.scaleLoader.scale_samples(X_valid, scaler_instance)
            
            # get validtiaon output
            if model_name == 'GBDT':
                y_valid_pred_result = model_handle.predict_proba(X_valid)
                y_valid_pred = y_valid_pred_result[:,0]
            elif model_name == 'DF':
                y_valid_pred = model_handle.predict(X_valid)
            else:
                print("Check model_name")

            # Update metric
            for met in metric_ftns:
                validation_metrics.update(met.__name__, met(y_valid_pred, y_valid_label))

            # Save output prediction array 
            save_output_file_name = result_dir / f'cv{fold_number}_results{iPram+1}_{patient_id}.npy'
            with open(save_output_file_name, 'wb') as fileHandle:
                np.save(fileHandle, y_valid_pred_result)

            # Save output images if needed
            if save_image:
                X_full = self.datasetLoader.load_full_data(patient_id)
                X_full = self.scaleLoader.scale_samples(X_full, scaler_instance)
                if model_name == 'GBDT':
                    y_pred_result = modelHandle.predict_proba(X_full)
                    y_pred_full = y_pred_result[:,0]
                elif model_name == 'DF':
                    y_pred_full = modelHandle.predict(X_full)

                image_save_name = result_dir / f'{patient_id}-{model_name.lower()}.png'

                img_out = np.reshape(y_pred, (512, 512))
                img_out = np.transpose(img_out)
                imsave(image_save_name, img_out)
        
        # Save the validation results
        validation_record_df = pd.DataFrame.from_records([validation_metrics.result()])
        validation_record_df.to_csv(save_file_name)
        
        return 0

def cross_validation(undersample_name, model_name, iFold, job_num):
    """
    Perform cross-validation on traning set with Parallel settings 
    """
    ## Construct the Trainer Class

    dirHandle = BaseDir()
    base_data_dir = dirHandle.dir

    infor_name = 'patient.csv'
    indexLoader = IndexLoader(base_data_dir, infor_name)
    
    if 'deep1' in undersample_name:
        feature_number = 22
    else:
        feature_number = 37

    datasetLoader = DatasetLoader(base_data_dir, undersample_name, feature_number)

    split_dict = {'NFOLD': 4, 'TRAIN': 3, 'VALID': 1, 'TEST': 1}
    trainer = Trainer(indexLoader, datasetLoader, split_dict)

    ######### RUN THE PARALLEL FUNCTION ###########
    print(f"Running the {iFold+1} fold")
    metric_df = trainer.init_model_result_saver(model_params_opts, iFold, metric_ftns)
    param_count = trainer.get_param_combination_count()

    # Ref: https://stackoverflow.com/questions/50528331/parallel-class-function-calls-using-python-joblib
    output = Parallel(verbose=10, n_jobs=job_num
                    )(delayed(trainer.train_across_iParm)(iPram, iFold, GradientBoostingClassifier, model_name, metric_ftns)
                        for iPram in range(param_count))
        
    # Save output on a fold level
    metric_dict = trainer.get_metric_dict()
    for iPram in range(param_count):
        _, result_dir = trainer.return_save_dir(iFold)
        save_file_name = result_dir / f'cv{iFold}_results{iPram+1}.csv'
        validation_df = pd.read_csv(result_dir / save_file_name)
        for met in metric_ftns:
            metric_df.loc[iPram, metric_dict[met.__name__]] = validation_df.loc[0,met.__name__]
        metric_df.to_csv(result_dir / f'cv{iFold}_results.csv')


if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--fold_index', default=0, type=int,
                      help='the fold on which the model would run')
    args.add_argument('-n', '--cc_number', default=4000, type=int,
                      help='The undersampling')    
    args.add_argument('-j', '--job_num', default=6, type=int,
                      help='The number of parallel jobs')        
    if not isinstance(args, tuple):
        # Use default value if there are no inputs
        args = args.parse_args()

    fold_index = args.fold_index
    cc_number = args.cc_number
    job_num = args.job_num

    # Define the name of the model
    model_name = 'GBDT'
    
    # The tuning params
    model_params_opts = {'learning_rate': [0.01, 0.05, 0.1],
         'n_estimators': [100, 500, 1000, 2000],
         'max_depth':[3, 5, 10, 20]} 

    # Define the metric that we want to evaluate on 
    metric_ftns = [precision, accuracy, sensitivity, specificity, f1_score, auroc, iou]
    
     
    ###################!!CHANGE!!##################
    # undersample_name = f'1_TL_CC_{cc_number}'
    undersample_name = '1_TL'    
    ###############################################

    print(f"  ------------------------Check HERE! ----------------------")
    print(f"  The undersampling folder you run on is {undersample_name}")

    cross_validation(undersample_name, model_name, fold_index, job_num)

