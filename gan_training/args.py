
from pathlib import Path

def training_args(hyper_params, batch_setting):
    
    all_kwags = {}

    # Define the generator model 
    modelG_kwags = {'type': None, 'args': {}}
    modelG_kwags['type'] = 'UNet'
    modelG_kwags['args']['n_channels'] = 2
    modelG_kwags['args']['n_classes'] = 1
    modelG_kwags['args']['depth'] = 2
    all_kwags['modelG_kwags'] = modelG_kwags

    # Define the discriminator model 
    modelD_kwags = {'type': None, 'args': {}}
    modelD_kwags['type'] = 'ResNet18'
    modelD_kwags['args']['input_channel'] = 1
    modelD_kwags['args']['output_dim'] = 2
    all_kwags['modelD_kwags'] = modelD_kwags

    # Define the generator input dataset 
    umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UoMR')
    ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UKR')
    dir_list = [umr_dir, ukr_dir]

    datasetG_kwags = {'type': None, 'args': {}}
    datasetG_kwags['type'] = 'GeneratorInput'
    datasetG_kwags['args']['dir_list'] = dir_list
    datasetG_kwags['args']['transform'] = 'None'
    datasetG_kwags['args']['file_name'] = 'stenosis_detail.csv'
    all_kwags['datasetG_kwags'] = datasetG_kwags

    # Define the discriminator input dataset  
    umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UMR\Full')
    ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UKR\Full')
    dir_list = [umr_dir, ukr_dir]

    datasetD_kwags = {'type': None, 'args': {}}
    datasetD_kwags['type'] = 'RealImage'
    datasetD_kwags['args']['dir_list'] = dir_list
    datasetD_kwags['args']['transform'] = 'None'
    datasetD_kwags['args']['file_name'] = 'image_infor.csv'
    all_kwags['datasetD_kwags'] = datasetD_kwags

    # Define the dataloader
    dataload_kwags = {'type': None, 'args': {}}
    dataload_kwags['type'] = 'DataLoader'
    dataload_kwags['args']['batch_size'] = batch_setting['batch_size']
    dataload_kwags['args']['num_workers'] = 0
    all_kwags['dataload_kwags'] = dataload_kwags


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

    # TODO: Not decided yet
    metric_kwags = {'type': None, 'args': {}}
    metric_kwags['type'] = 'MetricCalculator'
    metric_kwags['args']['target_metric'] = ['precision', 'auprc', 'auroc', 'sensitivity', 'specificity', 'f1']
    metric_kwags['args']['method'] = 3 # The 0309 Run used method 4
    all_kwags['metric_kwags'] = metric_kwags

    loss_kwags = {}
    loss_kwags['sigma_regularizer'] = 10
    all_kwags['loss_kwags'] = loss_kwags

    training_kwags = {}
    training_kwags['start_epoch'] = 1
    training_kwags['max_epoch'] = batch_setting['max_epoch'] ### CONSIDER ###
    training_kwags['early_stop'] = hyper_params['early_stop'] ### CONSIDER ###
    training_kwags['log_step'] = int(np.sqrt(data_kwags['args']['batch_size']))
    all_kwags['training_kwags'] = training_kwags

    monitor_kwags = {}
    monitor_kwags['mnt_mode'] ='min'
    monitor_kwags['mnt_metric'] = 'loss'
    monitor_kwags['mnt_best'] = np.inf if monitor_kwags['mnt_mode'] == 'min' else -np.inf
    monitor_kwags['save_period'] = 1
    all_kwags['monitor_kwags'] = monitor_kwags

    infor_kwags = {}
    infor_kwags['save_dir'] = ''
    infor_kwags['task_name'] = batch_setting['task_name'] 
    infor_kwags['log_date'] = batch_setting['date'] 
    infor_kwags['save_name'] = get_save_name_from_hyper_params(hyper_params)
    infor_kwags['checkpoint_dir'] = Path(infor_kwags['save_dir']) / 'model' / str(infor_kwags['log_date']) / infor_kwags['task_name'] / infor_kwags['save_name']
    infor_kwags['log_dir'] = Path(infor_kwags['save_dir']) / 'log' / str(infor_kwags['log_date']) / infor_kwags['task_name']
    infor_kwags['log_content'] = hyper_params
    infor_kwags['log_name'] = f"{infor_kwags['save_name']}.csv"
    infor_kwags['print_to_screen'] = batch_setting['print_to_screen']
    all_kwags['infor_kwags'] = infor_kwags