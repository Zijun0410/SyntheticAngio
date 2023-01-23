from re import S
from args import *
import argparse
import torch
import torch.nn as nn
import main_utils as utils
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import model as module_arch
import dataset as module_dataset
import torch.utils.data as module_dataloader
import torchvision.utils as vision_utils

import torch.optim as module_optim
import torch.optim.lr_scheduler as module_schedular
from itertools import cycle


def main(all_args):
    utils.fix_random_seeds()
    # Initiate the trainer
    trainer = Trainer(**all_args)
    trainer.train()

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, model_kwags, dataset_train_kwags, dataset_val_kwags,  
        optim_kwags, schedular_kwags, training_kwags, monitor_kwags, inforlog_kwags):

        # Decide the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # TODO: add the multi-gpu support
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        # Initiate the ojectve detection model
        model_handle = self.init_class(module_arch, model_kwags)
        self.model = model_handle.get_model().to(self.device)
        module_arch.count_parameters(self.model, 'FCOS')

        # Initiate the dataset
        self.train_data = self.init_class(module_dataset, dataset_train_kwags)
        self.val_data = self.init_class(module_dataset, dataset_val_kwags)
        
        # Initiate training logic attributes
        # batch_size, start_epoch, max_epoch, log_step, early_stop, pretrained_checkpoint
        self.init_attribute(training_kwags)

        # Initiate the dataloader
        dataloader_kwags = {'type': 'DataLoader', 'args': {}}
        dataloader_kwags['args'] = {'batch_size': self.batch_size,
                        'collate_fn':module_dataloader.dataloader.default_collate,
                        'drop_last':False}
        if not torch.cuda.is_available():
            dataloader_kwags['args'].update({'num_workers':0})

        dataloader_kwags['args'].update({'sampler':module_dataloader.sampler.RandomSampler(range(len(self.train_data)))})
        self.train_loader = self.init_class(module_dataloader, 
                dataloader_kwags, self.train_data)
        dataloader_kwags['args'].update({'sampler':module_dataloader.sampler.RandomSampler(range(len(self.val_data)))})
        self.val_loader = self.init_class(module_dataloader, 
                dataloader_kwags, self.val_data)

        # Initiate the optimizers
        trainable_params = filter(lambda p: p.requires_grad, 
                                self.model.parameters())
        self.optimizer = self.init_class(module_optim, 
            optim_kwags, trainable_params)

        # Initiate the learning rate schedular
        self.lr_scheduler = None
        if schedular_kwags:
            self.lr_scheduler = self.init_class(module_schedular, 
                schedular_kwags, self.optimizer)
        
        # Initiate loss monitoring attributes
        # mnt_mode, mnt_metric, mnt_best, save_period
        self.init_attribute(monitor_kwags)

        # Initiate the logging attribute: 
        # log_dir, checkpoint_dir, save_dir, output_dir
        self.init_attribute(inforlog_kwags)
        
        self.writer = SummaryWriter(self.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpt = self.load_checkpoint()

        if checkpt is not None:

            self.model.load_state_dict(checkpt['model'])
            self.optimizer.load_state_dict(checkpt['optimizer'])
            self.start_epoch = checkpt['epoch'] + 1

            try:
                self.lr_scheduler.load_state_dict(checkpt['scheduler'])
            except TypeError:
                print('Warning: there is a mismatch for the schedular type.')

    # def obtain_fix_batch_for_visualization(self):
    #     """
    #     Obtain fix batch for visualization
    #     """
    #     return torch.cat([self.generator_input_data[i].unsqueeze(0) 
    #                         for i in range(self.batch_size)], dim=0)

    def train(self):
        """
        Full training logic
        """
        print('----------- START TRAINING ------------')
        
        # fixed_batch_input = self.obtain_fix_batch_for_visualization().to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train(True)

            print(f"Training for the {epoch}th epochs")
            total_loss = 0.0

            for batch_idx, (targets, images) in enumerate(self.train_loader):

                images = [images.to(self.device) for image in images]
                targets = [target.to(self.device) for target in targets]

                self.optimizer.zero_grad()

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()

                # Update the optimizer 
                self.optimizer.step()
                
                # Record the total loss
                total_loss += losses.item()
                
                if batch_idx % self.log_step == 0:
                    # Log the scalar values
                    self.writer.add_scalar('batch_loss/total_loss', errD.item(), batch_count)
                    self.writer.add_scalar('batch_loss/generator loss', errG.item(), batch_count)
                    self.writer.add_scalar('batch_stats/D(x)', D_x, batch_count)
                    self.writer.add_scalar('batch_stats/ratio of D(G(z))', D_G_z1/D_G_z2, batch_count)
                    # Print out the log information
                    print(f"[{epoch + 1}/{self.max_epoch}][{batch_idx + 1}/{len(self.generator_loader)}] " +
                        f"Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} " + 
                        f"D(x): {D_x:.3f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.3f}")

                # # The image is saved every 1000 epoch.
                # if batch_count % 1000 == 0:
                #     fix_batch_synthetic_images = self.generator(fixed_batch_input).detach()
                #     vision_utils.save_image(fix_batch_synthetic_images,
                #                       self.log_dir / "output" / f"sythetic_samples_{batch_count}.png",
                #                       normalize=True)

            # Epoch Level Saving
            self.writer.add_scalar('loss/discriminator loss', loss_discriminator / len(self.generator_loader), epoch)
            if count_generator_update != 0:
                self.writer.add_scalar('loss/generator loss', loss_generator / count_generator_update, epoch)
            
            # Save the model checkpoints

            ## Checkpoint Creation
            checkpoint = dict()
            checkpoint['discriminator'] = self.discriminator.state_dict()
            checkpoint['generator'] = self.generator.state_dict()
            checkpoint['discriminator_optimizer'] = self.discriminator_optimizer.state_dict()
            checkpoint['generator_optimizer'] = self.generator_optimizer.state_dict()
            checkpoint['epoch'] = epoch
            torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_{epoch}.pt")

        self.writer.close()
        print('Finished Training')

    def init_attribute(self, kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def init_class(self, module, class_kwags, *args, **kwargs):
        module_name = class_kwags['type']
        module_args = dict(class_kwags['args'])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def load_checkpoint(self):
        """
        Load checkpoint if self.pretrained_checkpoint is not None
        or resume from self.checkpoint_dir 
        """
        checkpt = None
        if self.pretrained_checkpoint is None and os.path.isdir(self.checkpoint_dir):
            # Try to resume from the latest checkpoint if self.pretrained_checkpoint is None
            latest_ckpt_pth = utils.get_the_latest_ckpt(self.checkpoint_dir)
            if latest_ckpt_pth is not None:
                checkpt = torch.load(self.checkpoint_dir / latest_ckpt_pth)
                print(f'Load the latest checkpoint: {latest_ckpt_pth}')
            else:
                print(f'No checkpoint found in {self.checkpoint_dir}, the model is initiated without existing checkpoints.')
        elif self.pretrained_checkpoint is not None and (self.pretrained_checkpoint.endswith('.pt') 
            and os.path.isfile(self.pretrained_checkpoint)):
            # load the pretrained checkpoint if self.pretrained_checkpoint is not None
            print(f'Load pretrained checkpoint: {self.pretrained_checkpoint}')
            checkpt = torch.load(self.pretrained_checkpoint)
        else:
            print(f'The model is initiated without existing checkpoints and is saved to {self.checkpoint_dir}')
            self.ensure_dir(self.checkpoint_dir)
        return checkpt

    def ensure_dir(self, dirname):
        """
        Helper function: ensure that the given directory exist
        """
        dirname = Path(dirname)
        if not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()


    # # parser.add_argument('-d', '--dataset', default='rca', type=str,
    # #                   help='The dataset type, rca, lca or all')

    # args_input = parser.parse_args()

    # genrator_type_dict = dict(zip(['unet', ''], ['']))
    # generator_type = args_input.generator_type
    # disciminator_str = args_input.disciminator_type
    # disciminator_type_dict = dict(zip(['resnet18'], ['ResNet18']))
    # generator_depth = args_input.generator_depth
    # dataset = args_input.dataset
    # assert dataset in ['rca', 'lca', 'all']

    hyper_params = {}
    hyper_params['batch_size'] = 8
    hyper_params['learning_rate'] = 1e-4
    hyper_params['weight_decay'] = 1e-5
    hyper_params['max_epoch'] = 70
    hyper_params['lr_schedule'] = True
    hyper_params['early_stop'] = 15

    batch_setting = {}
    batch_setting['checkpoint_path'] = None
    batch_setting['task_name'] = 'DEBUG'
    batch_setting['date'] = 20230123

    all_kwags = training_args(hyper_params, batch_setting)

    main(all_kwags)