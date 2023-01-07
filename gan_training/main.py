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
    def __init__(self, modelG_kwags, modelD_kwags, datasetG_kwags, datasetD_kwags,  
        optim_kwags, schedular_kwags, training_kwags, monitor_kwags, inforlog_kwags):

        # Decide the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # TODO: add the multi-gpu support
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        # Initiate the generator and discriminator
        self.generator = self.init_class(module_arch, modelG_kwags).to(self.device)
        discriminator_handle = self.init_class(module_arch, modelD_kwags)
        self.discriminator = discriminator_handle.get_model().to(self.device)
        module_arch.count_parameters(self.generator, 'Generator')
        module_arch.count_parameters(self.discriminator, 'Discriminator')

        # TODO: weight initialization of the model 
        # self.generator.apply(utils.weights_init)
        # self.discriminator.apply(utils.weights_init)

        # Initiate the dataset
        self.generator_input_data = self.init_class(module_dataset, datasetG_kwags)
        self.discriminator_input_data = self.init_class(module_dataset, datasetD_kwags)
        
        
        # Initiate training logic attributes
        # batch_size, start_epoch, max_epoch, log_step, early_stop, pretrained_checkpoint, generator_step
        self.init_attribute(training_kwags)

        # Initiate the dataloader
        dataloader_kwags = {'type': 'DataLoader', 'args': {}}
        dataloader_kwags['args'] = {'batch_size': self.batch_size,
                        'collate_fn':module_dataloader.dataloader.default_collate,
                        'drop_last':True}
        if not torch.cuda.is_available():
            dataloader_kwags['args'].update({'num_workers':0})

        dataloader_kwags['args'].update({'sampler':module_dataloader.sampler.RandomSampler(range(len(self.generator_input_data)))})
        self.generator_loader = self.init_class(module_dataloader, 
                dataloader_kwags, self.generator_input_data)
        dataloader_kwags['args'].update({'sampler':module_dataloader.sampler.RandomSampler(range(len(self.discriminator_input_data)))})
        self.discriminator_loader = self.init_class(module_dataloader, 
                dataloader_kwags, self.discriminator_input_data)

        # Define the loss function
        self.criterion = nn.BCELoss()

        # Initiate the optimizers
        generator_trainable_params = filter(lambda p: p.requires_grad, 
                                            self.generator.parameters())
        discriminator_trainable_params = filter(lambda p: p.requires_grad, 
                                                self.discriminator.parameters())
        self.generator_optimizer = self.init_class(module_optim, 
            optim_kwags, generator_trainable_params)
        self.discriminator_optimizer = self.init_class(module_optim, 
            optim_kwags, discriminator_trainable_params)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

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

            self.generator.load_state_dict(checkpt['generator'])
            self.discriminator.load_state_dict(checkpt['discriminator'])
            self.start_epoch = checkpt['epoch'] + 1

            self.generator_optimizer.load_state_dict(checkpt['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(checkpt['discriminator_optimizer'])

            # try:
            #     scheduler.load_state_dict(checkpt['scheduler'])
            # except TypeError:
            #     print('Warning: there is a mismatch for the schedular type.')

    def obtain_fix_batch_for_visualization(self):
        """
        Obtain fix batch for visualization
        """
        return torch.cat([self.generator_input_data[i].unsqueeze(0) 
                            for i in range(self.batch_size)], dim=0)

    def train(self):
        """
        Full training logic
        """
        print('----------- START TRAINING ------------')
        
        fixed_batch_input = self.obtain_fix_batch_for_visualization().to(self.device)

        for epoch in range(self.start_epoch, self.max_epoch):
            self.generator.train(True)
            self.discriminator.train(True)

            print(f"Training for the {epoch}th epochs")

            loss_discriminator = 0.0
            loss_generator = 0.0
            count_generator_update = 0

            for batch_idx, (input_image, real_image) in enumerate(zip(
                self.generator_loader, cycle(self.discriminator_loader))):

                # input_image <torch.Size([batch_size, 3, 512, 512])>: input image for the generator
                input_images = input_image.to(self.device)
                # real_image <torch.Size([batch_size, 1, 512, 512])>: real image for the discriminator
                real_images = real_image.to(self.device)

                ##############################################
                # (1) Update D network: minimzie D(x) - D(G(z))
                # http://xxx.itp.ac.cn/pdf/1704.00028
                ##############################################

                # Set discriminator gradients to zero
                self.discriminator_optimizer.zero_grad()

                # Train with real images
                real_output = self.discriminator(real_images)
                errD_real = torch.mean(real_output)
                D_x = real_output.mean().item()

                # Generate synthetic image batch with generator
                synthetic_images  = self.generator(input_images)

                # Train with synthetic images
                synthetic_output = self.discriminator(synthetic_images.detach())
                errD_synthetic = torch.mean(synthetic_output)
                D_G_z1 = synthetic_output.mean().item()

                # Calculate W-div gradient penalty
                gradient_penalty = utils.calculate_gradient_penalty(self.discriminator,
                                        real_images.data, synthetic_images.data, self.device)

                # Add the gradients from the all-real and all-synthetic batches
                errD = - errD_real + errD_synthetic + gradient_penalty * 10
                errD.backward()

                # Update the discriminator
                self.discriminator_optimizer.step()
                
                # Record the discriminator loss
                loss_discriminator += errD.item()

                # Train the generator every generator_step iterations
                if (batch_idx + 1) % self.generator_step == 0:
                    
                    ##############################################
                    # (2) Update G network: minimize D(G(z))
                    ##############################################
                    # Set generator gradients to zero
                    self.generator_optimizer.zero_grad()

                    # Generate synthetic image batch with the generator
                    synthetic_images = self.generator(input_images)
                    synthetic_output = self.discriminator(synthetic_images)
                    errG = - torch.mean(synthetic_output)
                    D_G_z2 = synthetic_output.mean().item()

                    errG.backward()

                    # Update the generator
                    self.generator_optimizer.step()

                    # Record the generator loss
                    loss_generator += errG.item()
                    count_generator_update += 1

                batch_count = batch_idx + epoch * len(self.generator_loader) + 1
                
                if batch_count % self.log_step == 0:
                    # Log the scalar values
                    self.writer.add_scalar('batch_loss/discriminator loss', errD.item(), batch_count)
                    self.writer.add_scalar('batch_loss/generator loss', errG.item(), batch_count)
                    self.writer.add_scalar('batch_stats/D(x)', D_x, batch_count)
                    self.writer.add_scalar('batch_stats/ratio of D(G(z))', D_G_z1/D_G_z2, batch_count)
                    # Print out the log information
                    print(f"[{epoch + 1}/{self.max_epoch}][{batch_idx + 1}/{len(self.generator_loader)}] " +
                        f"Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} " + 
                        f"D(x): {D_x:.3f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.3f}")

                # The image is saved every 1000 epoch.
                if batch_count % 1000 == 0:
                    fix_batch_synthetic_images = self.generator(fixed_batch_input).detach()
                    vision_utils.save_image(fix_batch_synthetic_images,
                                      self.log_dir / "output" / f"sythetic_samples_{batch_count}.png",
                                      normalize=True)

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
    parser = argparse.ArgumentParser()

    # parser.add_argument('-gt', '--generator_type', default='unet', type=str,
    #                   help='The type of the generator')

    parser.add_argument('-gd', '--generator_depth', default=2, type=int,
                      help='The depth of the generator')

    parser.add_argument('-dt', '--disciminator_type', default='resnet18', type=str,
                      help='The type of the disciminator') 

    # parser.add_argument('-d', '--dataset', default='rca', type=str,
    #                   help='The dataset type, rca, lca or all')

    args_input = parser.parse_args()

    # genrator_type_dict = dict(zip(['unet', ''], ['']))
    # generator_type = args_input.generator_type
    disciminator_str = args_input.disciminator_type
    disciminator_type_dict = dict(zip(['resnet18'], ['ResNet18']))
    generator_depth = args_input.generator_depth
    # dataset = args_input.dataset
    # assert dataset in ['rca', 'lca', 'all']

    hyper_params = {}
    # batch_setting['generator_type'] = generator_type
    hyper_params['disciminator_type'] = disciminator_type_dict[disciminator_str]
    hyper_params['generator_depth'] = generator_depth
    hyper_params['batch_size'] = 8
    hyper_params['learning_rate'] = 1e-4
    hyper_params['max_epoch'] = 40
    hyper_params['generator_step'] = 1

    hyper_params['lr_schedule'] = False
    hyper_params['early_stop'] = 10

    batch_setting = {}
    batch_setting['checkpoint_path'] = None
    batch_setting['task_name'] = 'TRY_Group_2_INIT_36_CHANNEL_2'
    batch_setting['date'] = 20221220

    all_kwags = training_args(hyper_params, batch_setting)

    main(all_kwags)