import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from itertools import accumulate

from dataset import * 

class SytheticDataLoader(object):
    """

    """
    def __init__(self, data_dir, running_purpose, batch_identifier=None, input_channel='L', 
        augmentation_code='None', batch_size=8, num_workers=2, shuffle=False, collate_fn=default_collate):

        self.purpose = running_purpose
        augmentation_dict = dict(zip(['Nature', 'None', 'Customized'], 
            [nature_paper_augmentation, no_augmentation, custom_augmentation]))
        augmentation_apply = augmentation_dict[augmentation_code]

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle':shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        if self.purpose == 'training':
            # Construct two seperate datasets
            # data_dir, dataset_id, augmentation_apply, purpose, input_channel
            if batch_identifier is not None:
                batch_id = f'UoMR_{batch_identifier}'
            else:
                batch_id = 'UoMR'
            self.training_set = SytheticDataset(data_dir, batch_id, augmentation_apply, 'train')
            self.validation_set = SytheticDataset(data_dir, batch_id, augmentation_apply, 'validation') 
        elif self.purpose == 'testing':
            if batch_identifier is not None:
                batch_id = f'UKR_{batch_identifier}'
            else:
                batch_id = 'UKR'
            self.testing_set = SytheticDataset(data_dir, batch_id, augmentation_apply, 'test')
        else:
            print("Running purpose not defined.")

    def training_loader(self):
        self.init_kwargs['dataset'] = self.training_set
        # Construct a DataLoader instance with training
        return DataLoader(sampler=SubsetRandomSampler(self.training_set.get_target_index()), **self.init_kwargs)

    def validation_loader(self):
        self.init_kwargs['dataset'] = self.validation_set
        return DataLoader(sampler=SubsetRandomSampler(self.validation_set.get_target_index()), **self.init_kwargs)

    def testing_loader(self):
        self.init_kwargs['dataset'] = self.testing_set
        return DataLoader(sampler=SubsetRandomSampler(self.testing_set.get_target_index()), **self.init_kwargs)
    
    def get_save_dir(self, folder):
        return self.init_kwargs['dataset'].get_save_path(folder)


class RealDataLoader(object):
    """DataLoader for loading real XCA images"""
    def __init__(self, load_dir, save_dir, side='ALL', batch_size=8, num_workers=2, shuffle=False, collate_fn=default_collate):
        
        self.dataset = RealDataset(load_dir, save_dir, side, no_augmentation)
        self.save_dir = save_dir
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle':shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'dataset': self.dataset
        }

    def get_data_loader(self):
        return DataLoader(sampler=SubsetRandomSampler(self.dataset.get_target_index()), **self.init_kwargs)

    def get_dataset(self):
        return self.dataset

    def get_save_dir(self, folder):
        return Path(self.save_dir) / folder