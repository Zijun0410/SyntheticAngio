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

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders. Apart from the build-in Dataloader, 
    BaseDataLoader handle the spliting and rendering of validation set.
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, dataset_labels, 
        separate_dataset_lengths, validation_split_input, stratify_level, collate_fn=default_collate):
        
        if not isinstance(validation_split_input,list):
            self.validation_splits = [validation_split_input]
        else:
            self.validation_splits = validation_split_input

        self.shuffle = shuffle

        self._sample_number = len(dataset)

        # When there are more than one validation split scenarios that we need to consider
        # First, split the full index and full dataset lables by the number of samples (i.e.length) from each dataset
        index_full = np.arange(self._sample_number)
        splited_index = [index_full[x - y: x] for x, y in zip(accumulate(separate_dataset_lengths), 
            separate_dataset_lengths)] 
        # [[1,2,3],[4,5,6,7,8,9],[10,11,12]]
        splited_labels = [dataset_labels[x - y: x] for x, y in zip(accumulate(separate_dataset_lengths), 
            separate_dataset_lengths)]
        # [[L,L,R],[L,L,L,L,R,R],[R,R,R]]

        # Iterate through different dataset
        dataset_num = 0
        overall_train_idx, overall_valid_idx = [],[]
        for validation_split in self.validation_splits:
            dataset_specific_index = splited_index[dataset_num]
            dataset_specific_label = splited_labels[dataset_num]
            if isinstance(validation_split,float) or isinstance(validation_split,str):
                if stratify_level == 2:
                    temp_train_idx, temp_valid_idx = train_test_split(dataset_specific_index, 
                    test_size=validation_split, stratify=dataset_specific_label)
                elif stratify_level == 1:
                    temp_train_idx, temp_valid_idx = self._split_sampler_random(validation_split, 
                        dataset_specific_index)
                elif stratify_level == 0:
                    temp_train_idx, temp_valid_idx = self._split_sampler_unbalance(dataset_specific_label, 
                        dataset_specific_index, validation_split)
                else:
                    raise ValueError('The variable stratify_level should be 0, 1 or 2.')
                overall_train_idx += list(temp_train_idx)
                overall_valid_idx += list(temp_valid_idx)

            elif validation_split==0:
                # all index go into to the trainig set 
                overall_train_idx += list(dataset_specific_index)
            elif validation_split==1:
                # all index go into to the testing/validation set
                overall_valid_idx += list(dataset_specific_index)
            else:
                raise ValueError('The variable validation_split should be within the range of (0,1).')
            dataset_num += 1

        self.sampler, self.valid_sampler = SubsetRandomSampler(overall_train_idx),SubsetRandomSampler(overall_valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self._sample_number = len(overall_train_idx)

        # this attribute is used in trainer.py
        self.batch_size = batch_size

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        # Construct a DataLoader instance with train_sampler (SubsetRandomSampler)
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def training_loader(self):
        return DataLoader(sampler=self.sampler, **self.init_kwargs)

    def validation_loader(self):
        return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def _split_sampler_unbalance(self, dataset_specific_labels, dataset_specific_index, 
        labels_to_valid_set, validation_split=0.25):
        # labels_to_valid_set could be either a string or a list of labels that we want 
        # to assign to validation set, for example "R".
        if isinstance(labels_to_valid_set,str):
            temp_list = list()
            temp_list.append(labels_to_valid_set)
            labels_to_valid_set = temp_list
        if sum([label_to_valid_set in dataset_specific_labels for label_to_valid_set in 
            labels_to_valid_set]) == len(labels_to_valid_set):
            train_idx, valid_idx = [], []
            for index, dataset_label in enumerate(dataset_specific_labels):
                if dataset_label in labels_to_valid_set and (len(valid_idx)/len(dataset_specific_index)) <= validation_split:
                    # First, it must be the required lable for validation set, e.g. "R"
                    # Second, the number of samples in validation should not surpass the 
                    #         validation split, and the default validation split is 0.25.
                    valid_idx.append(dataset_specific_index[index])
                else:
                    train_idx.append(dataset_specific_index[index])
            return train_idx, valid_idx
        else:
            raise ValueError('Label {} not found in labels of the dataset.'.format(label_to_valid_set))

    def _split_sampler_random(self, split, idx_to_split):
        np.random.seed(0)
        np.random.shuffle(idx_to_split)
        if isinstance(split, int):
            assert split > 0
            assert split < self._sample_number, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self._sample_number * split)

        valid_idx = idx_to_split[0:len_valid]
        train_idx = np.delete(idx_to_split, np.arange(0, len_valid))
        return train_idx, valid_idx

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            # Construct a DataLoader instance with valid_sampler (SubsetRandomSampler)
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
    
    @property
    def sample_number(self):
        return self._sample_number

DATASET_NAMES = ['Alberto_Dataset','Public_Dataset']
DATASET_CLASS_HANDLE = [Alberto_Dataset, Public_Dataset]

class CombineDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, input_channel='L', dataset_names_by_user = DATASET_NAMES, 
        augmentation_code = 'Nature', shuffle=True, validation_split=[0.33, 0.33], num_workers=1, 
        label_split=True, stratify_level=2, training=True):
        
        # Arguments:
        #     data_dir (str): the directory where the data would be loaded
        #     batch_szie (int): size of the mini-batch
        #     input_channel (char): 'RGB' or 'L'
        #     dataset_names_by_user (list of strings): must be folder names that could found under the data_dir
        #                                 e.g.['Alberto_Dataset','Private_Database','Public_Dataset']
        #     validation_split (float\str, or list of floot\str): the percentage of validation split of the datasets.
        #            if stratify_level = 0, it could be string that indicate the lable we want to put into validation
        #     num_workers
        #     label_split (bool, or list of bool): whether or not to obtain the label of every image in the dataset.
        #     stratify_level (int, or list of int): 
        #           stratify_level = 2 : apply stratify_level validation split when constructing the sampler
        #           stratify_level = 1 : apply random validation split when constructing the sampler
        #           stratify_level = 0 : apply validation split in which training and validation contain  
        #                          mutual exclusive sample labels 
        
        # Make sure the validation split has same length with dataset_names_by user
        assert_infro = "The number of the dataset names given should match with that of the validation_split."
        assert (len(validation_split)==len(dataset_names_by_user) or len(validation_split)==1),assert_infro

        # Augmentation
        if augmentation_code == 'Nature':
            augmentation_apply = nature_paper_augmentation
        elif augmentation_code == 'Customized':
            augmentation_apply = custom_augmentation 
        else:
            raise KeyError('No augmentation method defined as {}.'.format(augmentation_code)) 

        # Match dataset_names with Class handle
        dataset_class_handles = dict(zip(DATASET_NAMES, DATASET_CLASS_HANDLE)) 

        self.dataset = CombinedDataset(data_dir, dataset_names_by_user, dataset_class_handles, 
            input_channel, augmentation_apply, label_split, training)
        
        dataset_labels = self.dataset.get_label()
        dataset_lengths = self.dataset.get_dataset_length()

        # If label_split = True, then the 'label' got here would be a list of length bigger than 0
        # Otherwise 'label' got here would be an empty list
        
        super().__init__(self.dataset, batch_size, shuffle, num_workers, dataset_labels, 
            dataset_lengths, validation_split, stratify_level)
        
        # The validation split and Sampler construction would be finished at class initializtion

