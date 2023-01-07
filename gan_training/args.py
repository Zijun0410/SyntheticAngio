from pathlib import Path
from main_utils import get_save_name_from_hyper_params
import numpy as np

def training_args(hyper_params, batch_setting):
    
    all_kwags = {}

    # Define the generator model 
    modelG_kwags = {'type': None, 'args': {}}
    modelG_kwags['type'] = 'UNet'
    modelG_kwags['args']['n_channels'] = 2
    modelG_kwags['args']['n_classes'] = 1
    modelG_kwags['args']['depth'] = hyper_params['generator_depth']
    modelG_kwags['args']['init_n'] = 36
    modelG_kwags['args']['group'] = 2
    all_kwags['modelG_kwags'] = modelG_kwags

    # Define the discriminator model 
    modelD_kwags = {'type': None, 'args': {}}
    modelD_kwags['type'] = hyper_params['disciminator_type']
    modelD_kwags['args']['input_channel'] = 1
    modelD_kwags['args']['output_dim'] = 2
    all_kwags['modelD_kwags'] = modelD_kwags

    # Define the generator input dataset 
    umr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UoMR')
    ukr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UKR')
    dir_list = [umr_dir, ukr_dir]

    datasetG_kwags = {'type': None, 'args': {}}
    datasetG_kwags['type'] = 'GeneratorInput'
    datasetG_kwags['args']['dir_list'] = dir_list
    datasetG_kwags['args']['transform'] = 'None'
    datasetG_kwags['args']['file_name'] = 'stenosis_detail.csv'
    all_kwags['datasetG_kwags'] = datasetG_kwags

    # Define the discriminator input dataset  
    umr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UMR\Full')
    ukr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UKR\Full')
    dir_list = [umr_dir, ukr_dir]

    datasetD_kwags = {'type': None, 'args': {}}
    datasetD_kwags['type'] = 'RealImage'
    datasetD_kwags['args']['dir_list'] = dir_list
    datasetD_kwags['args']['transform'] = 'None'
    datasetD_kwags['args']['file_name'] = 'image_infor.csv'
    all_kwags['datasetD_kwags'] = datasetD_kwags

    # # Define the dataloader
    # dataload_kwags = {'type': None, 'args': {}}
    # dataload_kwags['type'] = 'DataLoader'
    # dataload_kwags['args']['batch_size'] = hyper_params['batch_size']
    # dataload_kwags['args']['num_workers'] = 0
    # all_kwags['dataload_kwags'] = dataload_kwags

    # Define the optimizer
    optim_kwags = {'type': None, 'args': {}}
    optim_kwags['type'] = 'Adam'
    optim_kwags['args']['lr'] = hyper_params['learning_rate']
    optim_kwags['args']['weight_decay'] = 0
    all_kwags['optim_kwags'] = optim_kwags

    # Define the learning rate scheduler
    if hyper_params['lr_schedule']:
        schedular_kwags = {'type':None, 'args': {}}
        schedular_kwags['type'] = 'ReduceLROnPlateau'
        schedular_kwags['args']['factor'] = 0.5
        schedular_kwags['args']['patience'] = hyper_params['early_stop']-1
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
    training_kwags['log_step'] = int(np.sqrt(hyper_params['batch_size']))
    training_kwags['generator_step'] = hyper_params['generator_step']
    all_kwags['training_kwags'] = training_kwags

    # Define the monitor setting
    monitor_kwags = {}
    monitor_kwags['mnt_mode'] ='min'
    monitor_kwags['mnt_metric'] = 'loss'
    monitor_kwags['mnt_best'] = np.inf if monitor_kwags['mnt_mode'] == 'min' else -np.inf
    monitor_kwags['save_period'] = 1
    all_kwags['monitor_kwags'] = monitor_kwags

    # Define the log saving setting
    inforlog_kwags = {}
    inforlog_kwags['save_dir'] = ''
    task_name = batch_setting['task_name'] 
    log_date = batch_setting['date'] 
    save_name = get_save_name_from_hyper_params(hyper_params)
    inforlog_kwags['checkpoint_dir'] = Path(inforlog_kwags['save_dir']) / task_name / str(log_date) / save_name / 'model' 
    inforlog_kwags['output_dir'] = Path(inforlog_kwags['save_dir']) / task_name / str(log_date) / save_name / 'output' 
    inforlog_kwags['log_dir'] = Path(inforlog_kwags['save_dir']) / task_name / str(log_date) / save_name
    # inforlog_kwags['meta_content'] = hyper_params
    # inforlog_kwags['print_to_screen'] = batch_setting['print_to_screen']
    all_kwags['inforlog_kwags'] = inforlog_kwags

    return all_kwags
