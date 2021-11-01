from indexloader import IndexLoader
from scaleloader import ScaleLoader
from baseclass import BaseDir
from datasetloader import DatasetLoader

import pickle
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


    For Dataset Loader, the indicators are with ['nontest', 'test', 'train', 'validation'] or
                                           ['fold_train', 'fold_test'] + fold number or 'infer'

    For Index Loader, the indicators are [nontest, test, train, or validation] or fold number

    For Scale Loader, the indicators should either be string within either ['validation','testing','inference'] 
                                           or a integer that stands for one of the fold number

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
        elif dataset_indicator in ['validation','testing','inference']:
            model_dir = model_dir / f'{dataset_indicator}'
            result_dir = result_dir / f'{dataset_indicator}' 
        else:
            print('Warning when creating the directory for model&re saving ')          

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

    def get_metric_dict(self):
            # Purpose:
            #   helper function 
            return self.metric_dict

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

    def train_across_folds(self, fold_number, model_class, model_name, metric_ftns, metric_df, 
        save_image=False, rerun_flag=False, skip_flag=False):
        # Purpose:
        #    Perform Cross Validation for hyper-param tuning 
        # Param:
        #    model_class: a class instance of the target model
        #    model_name(string): the name of the target model
        #    fold_number(int): the fold that we want the training happen
        #    metric_ftns(list); a list of metric function 

        _, result_dir = self.return_save_dir(fold_number)

        if os.path.isfile(result_dir / f'cv{fold_number}_results.csv') and not rerun_flag:
            metric_df = pd.read_csv(result_dir / f'cv{fold_number}_results.csv')
            return metric_df
        
        for iPram in range(len(self.param_combination)):
            save_file_name = result_dir / f'cv{fold_number}_results{iPram+1}.csv'

            if os.path.isfile(save_file_name) and not rerun_flag:
                # If rerun_flag is false, then the result dataframe would directly be loaded instead of rerun
                validation_df = pd.read_csv(save_file_name)
                for met in metric_ftns:
                    metric_df.loc[iPram, self.metric_dict[met.__name__]] = validation_df.loc[0,met.__name__]
                print(f"The {iPram}th param is already done.")
                continue

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
                if skip_flag:
                    continue
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
                    # y_valid_pred = y_valid_pred_result[:,0]
                    y_valid_pred = y_valid_pred_result[:,1] # 1_TL
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
           
            validation_dict = validation_metrics.result()

            for met in metric_ftns:
                metric_df.loc[iPram, self.metric_dict[met.__name__]] = validation_dict[met.__name__]
           
            validation_record_df = pd.DataFrame.from_records([validation_dict])
            validation_record_df.to_csv(save_file_name)
        
        metric_df.to_csv(result_dir / f'cv{fold_number}_results.csv')
        return metric_df

def cross_validation(trainer, undersample_name, model_name, model_params_opts, metric_ftns): 

    for iFold in range(split_dict['NFOLD']):
        print(f"Running the {iFold+1} fold")
        metric_df = trainer.init_model_result_saver(model_params_opts, iFold, metric_ftns)
        metric_df = trainer.train_across_folds(iFold, GradientBoostingClassifier, model_name, 
            metric_ftns, metric_df,rerun_flag=True, skip_flag=True)
        if iFold==0:
            final_result_df = metric_df.copy()
        else:
            final_result_df = pd.concat([final_result_df,metric_df])

    _, result_dir = trainer.return_save_dir('random_string')

    final_result_df.to_csv(result_dir / f'cv_results.csv')



def hyper_param_selection(trainer, undersample_name, model_params_opts, metric_ftns):

    _, result_dir = trainer.return_save_dir('random_string')
    metric_dict  = trainer.get_metric_dict()

    reslut_df = pd.read_csv(result_dir / 'cv_results.csv')
    param_keys = model_params_opts.keys()
    param_list = list(itertools.product(*list(model_params_opts.values())))
    mean_series = []
    std_series = []
    for param_combo in param_list:
        target_param = dict(zip(param_keys,param_combo))
        reslut_df_sub = reslut_df[np.logical_and.reduce([reslut_df[k] == v for k,v in target_param.items()])] 
        mean_series.append(reslut_df_sub.mean(axis=0))
        std_series.append(reslut_df_sub.std(axis=0))
    mean_df = pd.DataFrame(mean_series)
    std_df = pd.DataFrame(std_series)
    std_df.rename(columns = dict(zip(list(reslut_df.columns), [f'{name}_std' for name in list(reslut_df.columns)])), inplace = True)

    organized_results = pd.concat( [mean_df, std_df[[f'{metric_dict[func_.__name__]}_std' for func_ in metric_ftns]]], axis=1)
    organized_results.drop([names for names in list(reslut_df.columns) if "Unnamed" in names], axis=1, inplace=True)
    ### TODOL the target is AUROC
    best_index = organized_results.idxmax(axis = 0).AUROC
    # print(best_index, param_list[int(best_index)], param_keys)
    best_param = dict(zip(param_keys,param_list[int(best_index)]))

    organized_results.to_csv(result_dir / f'cv_results_organised.csv')
    
    return best_param


def test(trainer, undersample_name, model_name, model_params, metric_ftns, save_image=True):

    # This is the major function for training
    # Param:
    #    undersample_name (string): the name of the folder where the undersampling results were saved
    #    model_name (string): GBDT or DF

    dataset_indicator = 'testing'
    train_set_casename = 'nontest'
    test_set_casename = 'test'

    scaler_instance = trainer.load_scaler(dataset_indicator,'stand')

    X_train, y_train = trainer.datasetLoader.load_undersampled_data(trainer.indexLoader, train_set_casename)
    
    X_train = trainer.scaleLoader.scale_samples(X_train, scaler_instance)


    model_name_save = f"{model_name}"
    model_trained = trainer.load_model(model_name_save, dataset_indicator)

    if model_name == 'GBDT':        
        print('Start GBDT training ....')
        if 'folds' in model_params: del model_params['folds']
        print(model_params)
        model_params['verbose'] = 1
        model_handle = GradientBoostingClassifier(**model_params)
    elif model_name == 'DF':
        model_params = {'random_state': 42}
        print('Start DF training ....')
        model_handle = CascadeForestClassifier(**model_params)
    else:
        print('Model not defined')
        return None

    if model_trained is None:
        kwargs = {"modelHandle":model_handle,"model_name":model_name_save, "dataset_indicator":dataset_indicator}
        model_handle = trainer.train(X_train, y_train, **kwargs)
        trainer.save_model(model_handle, model_name_save, dataset_indicator)  
    else:
        model_handle = model_trained  

    ## 4. Init MetricTracker
    test_metrics = MetricTracker(*[m.__name__ for m in metric_ftns])
    test_metrics.reset()

    target_patient_idx, _, _= trainer.datasetLoader.init_target_id_and_file_name(trainer.indexLoader, test_set_casename)

    _, result_dir = trainer.return_save_dir(dataset_indicator)

    for idx in target_patient_idx:
        print(f"Trainer Patientwise test: Runing the {idx}th case.")

        # Load the unsupervised model for performance comparison with cross-validation
        X_test, y_test_label = trainer.datasetLoader.load_unsupervised_data(idx)
        X_test = trainer.scaleLoader.scale_samples(X_test, scaler_instance)

        if model_name == 'GBDT':
            ## TODO: check when it gives the correct (& reverse) prediction
            y_test_pred_result = model_handle.predict_proba(X_test)
            y_test_pred = y_test_pred_result[:,0]
            # y_test_pred = y_test_pred_result[:,1] # 1_TL
        elif model_name == 'DF':
            y_test_pred = model_handle.predict(X_test)
        else:
            print('Check Model Name!')

        # Save output prediction array 
        save_output_file_name = result_dir / f'{idx}.npy'
        with open(save_output_file_name, 'wb') as fileHandle:
            np.save(fileHandle, y_test_pred_result)

        # Save output images if needed
        if save_image:
            X_full, y_full = trainer.datasetLoader.load_full_data(idx)
            X_full = trainer.scaleLoader.scale_samples(X_full, scaler_instance)
            if model_name == 'GBDT':
                y_pred_result = model_handle.predict_proba(X_full)
                y_pred_full = y_pred_result[:,0]
                # y_pred_full = y_pred_result[:,1] # 1_TL
            elif model_name == 'DF':
                y_pred_full = model_handle.predict(X_full)

            image_save_name = result_dir / f'{idx}-{model_name.lower()}.png'

            img_out = np.reshape(y_pred_full, (512, 512))
            img_out = np.transpose(img_out)
            imsave(image_save_name, img_out)

        # Update metric
        for met in metric_ftns:
            test_metrics.update(met.__name__, met(y_test_pred, y_test_label))

    test_df  = pd.DataFrame([test_metrics.result(), test_metrics.std()])

    test_df.to_csv(result_dir / f'test_results.csv')

def train_infer(trainer, undersample_name, model_name, model_params):

    dataset_indicator = 'inference'
    train_set_casename = 'infer'

    scaler_instance = trainer.load_scaler(dataset_indicator,'stand')

    X_train, y_train = trainer.datasetLoader.load_undersampled_data(trainer.indexLoader, train_set_casename)
    
    X_train = trainer.scaleLoader.scale_samples(X_train, scaler_instance)

    model_name_save = f"{model_name}"
    model_trained = trainer.load_model(model_name_save, dataset_indicator)

    if model_name == 'GBDT':        
        print('Start GBDT training ....')
        if 'folds' in model_params: del model_params['folds']
        print(model_params)
        model_params['verbose'] = 1
        model_handle = GradientBoostingClassifier(**model_params)
    else:
        print('Model not defined')
        return None

    if model_trained is None:
        kwargs = {"modelHandle":model_handle,"model_name":model_name_save, "dataset_indicator":dataset_indicator}
        model_handle = trainer.train(X_train, y_train, **kwargs)
        trainer.save_model(model_handle, model_name_save, dataset_indicator)  
    else:
        model_handle = model_trained  


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('-n', '--cc_number', default=4000, type=int,
                      help='The undersampling')  
    if not isinstance(args, tuple):
        # Use default value if there are no inputs
        args = args.parse_args()
    cc_number = args.cc_number

    ###################!!CHANGE!!##################
    # undersample_name = f'1_TL_CC_{cc_number}'
    # undersample_name = '1_TL'    
    undersample_name = '0_unsupervised_deep1'
    ###############################################

    print(f"  ------------------------Check HERE! ------------------------")
    print(f"  The undersampling folder you run on is {undersample_name}")
    print(f"  ------------------------Check HERE! ------------------------")

    model_name = 'GBDT'
    model_params_opts = {'learning_rate': [0.01, 0.05, 0.1],
         'n_estimators': [100, 500, 1000, 2000],
         'max_depth':[3, 5, 10, 20]} 

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

    metric_ftns = [precision, accuracy, sensitivity, specificity, f1_score, auroc, iou]
    
    # 1. Cross Validation
    # HERE we assume that the cross-validtaion process has already been done by 
    #  the corss-validation fuunction in train_par.py

    # The crosss-validation function in the train.py is super slow, given by 
    # cross_validation(trainer, undersample_name, model_name, model_params_opts, metric_ftns)

    # 2. Hyper-param Selection
    # best_param = hyper_param_selection(trainer, undersample_name, model_params_opts, metric_ftns)
    # Instead of selecting best_param with corss-valdiation we can also set the best hyper param
    # for the final training and testing

    best_param = {'learning_rate': 0.05,
         'n_estimators': 1000,
         'max_depth':10}

    # 3. Test
    # test(trainer, undersample_name, model_name, best_param, metric_ftns)

    # 4. Train the inference model
    train_infer(trainer, undersample_name, model_name, best_param)
