import argparse
import collections
import torch
import numpy as np
import base.base_data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import EasyTrainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def training(config):
    # setup logger 
    logger = config.get_logger('train')

    # breakpoint()
    # construct a DataLoader instance with a SubsetRandomSampler instance(build upon all the 
    # training indices) by passing in class  handle and the args from the config.

    data_loader = config.init_obj('data_loader', module_data)
    
    # construct the training and validation DataLoader instance
    triain_data_loader = data_loader.training_loader()
    valid_data_loader = data_loader.validation_loader()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = EasyTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=triain_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    ##################################### Testing ####################################

    if config['data_loader']['args']['running_purpose'] == 'training':
        # if it's for model preparation purpose then there is no need for testing
        config['data_loader']['args']['running_purpose'] = 'testing'
        # data_loader = getattr(module_data, config['data_loader']['type'])(**)
        data_loader = config.init_obj('data_loader', module_data)
        

        trainer.test(module_arch, data_loader)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-bg', '--debug', default=None, type=str,
                      help='if run in a debug mode (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    # prase the argument, store hyperparameter in Dicts, 
    # set up all the directory for saving checkpoints and log
    # store the functional or class handle object 
    config = ConfigParser.from_args(args, options) # config.config is a Dict
    training(config)
