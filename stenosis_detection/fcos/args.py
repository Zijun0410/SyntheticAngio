from pathlib import Path
import main_utils as utils
import numpy as np

def training_args(hyper_params, batch_setting):
    
    all_kwags = {}

    # Define the discriminator model 
    model_kwags = {'type': None, 'args': {}}
    model_kwags['type'] = 'CustomizedFCOS'
    model_kwags['args']['num_class'] = 2
    model_kwags['args']['min_size'] = 512
    model_kwags['args']['max_size'] = 512
    all_kwags['model_kwags'] = model_kwags

    # Define the synthetic dataset 
    umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\UoMR')
    ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\UKR')

    dataset_train_kwags = {'type': None, 'args': {}}
    dataset_train_kwags['type'] = 'SyntheticImage'
    dataset_train_kwags['args']['dir_list'] = [umr_dir]
    dataset_train_kwags['args']['transform'] = 'None'
    dataset_train_kwags['args']['file_name'] = 'stenosis_detail.csv'
    all_kwags['dataset_train_kwags'] = dataset_train_kwags

    dataset_val_kwags = {'type': None, 'args': {}}
    dataset_val_kwags['type'] = 'SyntheticImage'
    dataset_val_kwags['args']['dir_list'] = [ukr_dir]
    dataset_val_kwags['args']['transform'] = 'None'
    dataset_val_kwags['args']['file_name'] = 'stenosis_detail.csv'
    all_kwags['dataset_val_kwags'] = dataset_val_kwags

    umr_test_dir = Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UMR\Full')

    dataset_test_kwags = {'type': None, 'args': {}}
    dataset_test_kwags['type'] = 'RealImage'
    dataset_test_kwags['args']['dir_list'] = [umr_test_dir]
    dataset_test_kwags['args']['transform'] = 'None'
    dataset_test_kwags['args']['file_name'] = 'image_infor.csv'
    dataset_test_kwags['args']['name'] = 'UMR'
    all_kwags['dataset_test_kwags'] = dataset_test_kwags

    # Define the optimizer
    optim_kwags = {'type': None, 'args': {}}
    optim_kwags['type'] = 'SGD'
    optim_kwags['args']['lr'] = hyper_params['learning_rate']
    optim_kwags['args']['weight_decay'] = hyper_params['weight_decay']
    optim_kwags['args']['nesterov'] = True
    optim_kwags['args']['momentum'] = 0.9
    all_kwags['optim_kwags'] = optim_kwags

    # Define the learning rate scheduler
    if hyper_params['lr_schedule']:
        schedular_kwags = {'type':None, 'args': {}}
        schedular_kwags['type'] = 'ReduceLROnPlateau'
        schedular_kwags['args']['factor'] = 0.5
        schedular_kwags['args']['patience'] = 10
        schedular_kwags['args']['min_lr'] = 1e-6
        schedular_kwags['args']['verbose'] = True
        all_kwags['schedular_kwags'] = schedular_kwags
    else:
        all_kwags['schedular_kwags'] = None

    # Define the training setting
    training_kwags = {}
    training_kwags['start_epoch'] = 1
    training_kwags['batch_size'] = hyper_params['batch_size']
    training_kwags['max_epoch'] = hyper_params['max_epoch'] ### CONSIDER ###
    training_kwags['early_stop'] = hyper_params['early_stop'] ### CONSIDER ###
    training_kwags['pretrained_checkpoint'] = batch_setting['checkpoint_path'] ### Default: None ###
    training_kwags['log_step'] = 20
    all_kwags['training_kwags'] = training_kwags

    # Define the monitor setting
    monitor_kwags = {}
    monitor_kwags['mnt_mode'] ='min'
    monitor_kwags['mnt_metric'] = 'val_loss'
    monitor_kwags['mnt_best'] = np.inf if monitor_kwags['mnt_mode'] == 'min' else -np.inf
    monitor_kwags['save_period'] = 1
    all_kwags['monitor_kwags'] = monitor_kwags

    # Define the log saving setting
    inforlog_kwags = {}
    inforlog_kwags['save_dir'] = ''
    task_name = batch_setting['task_name'] 
    log_date = batch_setting['date'] 
    save_name = utils.get_save_name_from_hyper_params(hyper_params)
    inforlog_kwags['checkpoint_dir'] = Path(inforlog_kwags['save_dir']) / str(log_date) / task_name  / 'model' 
    inforlog_kwags['output_dir'] = Path(inforlog_kwags['save_dir']) / str(log_date) / task_name / 'output' 
    inforlog_kwags['log_dir'] = Path(inforlog_kwags['save_dir']) / str(log_date) / task_name 
    # inforlog_kwags['meta_content'] = hyper_params
    # inforlog_kwags['print_to_screen'] = batch_setting['print_to_screen']
    all_kwags['inforlog_kwags'] = inforlog_kwags

    return all_kwags
