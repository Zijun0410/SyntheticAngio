from args import *
import argparse
import torch
import numpy as np
import main_util as utils
from pathlib import Path

import model as module_arch
import dataset as module_dataset
import torch.utils.data as module_dataloader

import torch.optim as module_optim
import torch.optim.lr_scheduler as module_schedular

def main(args):
    utils.fix_random_seeds()

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, modelG_kwags, modelD_kwags, datasetG_kwags, datasetD_kwags, dataloader_kwags, 
    optim_kwags, schedular_kwags, metric_kwags, loss_kwags, training_kwags, 
    monitor_kwags, infor_kwags):

        # Decide the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initiate the generator and discriminator
        self.generator = self.init_class(module_arch, modelG_kwags).to(self.device)
        self.discrminator = self.init_class(module_arch, modelD_kwags).to(self.device)
        # TODO: add the option to load the pretrained model
        # TODO: Print the number of model parameters
        # TDDO: weight initialization of the model 
        # from main_utils import weights_init
        # self.generator.apply(weights_init)
        # self.discrminator.apply(weights_init)

        # Initiate the dataset
        self.generator_input_data = self.init_class(module_dataset, datasetG_kwags)
        self.discrminator_input_data = self.init_class(module_dataset, datasetD_kwags)

        # Initiate the dataloader
        dataloader_kwags.update({'sampler': module_dataloader.sampler.RandomSampler,
                        'collate_fn':module_dataloader.dataloader.default_collate,
                        'drop_last':True})
        if not torch.cuda.is_available():
            dataloader_kwags.update({'num_workers':0})

        self.generator_loader = self.init_class(module_dataset, dataloader_kwags, self.generator_input_data)
        self.discrminator_loader = self.init_class(module_dataset, dataloader_kwags, self.discrminator_input_data)

        # Define the loss function
        # TODO

        # Initiate the optimizers
        # TODO: Should be two optimizers for generator and discriminator separately
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.init_class(module_optim, optim_kwags, trainable_params)

        # Initiate the learning rate schedular
        self.lr_scheduler = None
        if schedular_kwags:
            self.lr_scheduler = self.init_class(module_schedular, schedular_kwags, self.optimizer)
        
        # TODO: Decide the remaining attributes


    def init_attribute(self, kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def init_class(self, module, class_kwags, *args, **kwargs):
        module_name = class_kwags['type']
        module_args = dict(class_kwags['args'])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def ensure_dir(self, dirname):
        """
        Helper function: ensure that the given directory exist
        """
        dirname = Path(dirname)
        if not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=True)
    
    def train_generator(self):
        # TODO
        pass

    def train_discriminator(self):
        # TODO
        pass


if __name__ == '__main__':
    args = argparse.ArgumentParser('GAN')

    args.add_argument('-gt', '--generator_type', default='unet', type=str,
                      help='The type of the generator')
    args.add_argument('-gd', '--generator_depth', default=3, type=int,
                      help='The depth of the generator')

    args.add_argument('-dt', '--disciminator_type', default='resnet18', type=str,
                      help='The type of the disciminator') 

    args.add_argument('-d', '--dataset', default='all', type=str,
                      help='The dataset type')

    # args.add_argument('-a', '--availability', default=1.0, type=float,
    #                   help='The availability of priviledged infor') 
    # args.add_argument('-s', '--seed', default=1, type=int,
    #                   help='The batch index')

    parser = argparse.ArgumentParser('BYOL', parents=[moswl_utils.get_args_parser()])
    args = parser.parse_args()

    main()