import os,sys
import fnmatch
from PIL import Image
import numpy as np
from pathlib import Path
from pandas import read_csv
import glob
import torch
import torch.utils.data as D
from sklearn.model_selection import train_test_split
## ###################################################
# When creating a new DataSet Class
# These functions need to be implement

    # -------- Part of BaseDataLoader --------
    # def __getitem__(self, index):
    #     return None

    # def __len__(self):
    #     return None

    # -------- Used in CombineDataLoader --------
    # def get_label(self):
    #     return None

    # def get_dataset_length(self):
    #     return None

    # ----------- Used to Save Output ---------

## ##################################################

from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class SytheticDataset(D.Dataset):
    """
    This is for the sythetic data training and validation 
    """    
    def __init__(self, data_dir, dataset_id, augmentation_apply, purpose, input_channel='L'):
        """
        Inputs:
            data_dir: the Sythetic Output Folder
            augmentation_apply: different augumentation that could be applied before returning the image
        """
        self.base_dir = Path(data_dir) # Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\
        self.load_dir = self.base_dir / 'Sythetic_Output' / dataset_id # Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\{batch_id}
        self.infor_summary_df = read_csv(self.load_dir / 'infor.csv')

        self.input_channel = input_channel
        self.augmentation = augmentation_apply

        # The index starts count from 0
        if purpose == 'train':
            self.target_index = self.infor_summary_df.index[self.infor_summary_df['train']==1].tolist()   
        elif purpose == 'validation':
            self.target_index = self.infor_summary_df.index[self.infor_summary_df['train']==0].tolist()
        elif purpose == 'test':
            self.target_index = self.infor_summary_df.index.tolist()

        self.infor_df = self.infor_summary_df.iloc[self.target_index,:]
        self.name_list = self.infor_summary_df['output_folder'].tolist()

    def __getitem__(self, index):
        folder_name = self.name_list[index].split('+')[-1]
        image_raw = Image.open(self.load_dir / folder_name / f'synthetic_1.png').convert("L")
        image_seg = Image.open(self.load_dir / folder_name / f'segmentation.png').convert("L")

        transformed_raw, transformed_seg = transform_image(image_raw, image_seg, self.input_channel, self.augmentation)

        return index, transformed_raw, transformed_seg, folder_name

    def __len__(self):
        return len(self.infor_df)

    def get_save_path(self, folder):
        # Used in testing image saving for Trainer
        return self.base_dir / folder 

    def get_target_index(self):
        # Used in Data Loader Construction
        return list(range(len(self.infor_df)))


class RealDataset(D.Dataset):
    """
    This is a real Dataset for testing purpose   
    """    
    def __init__(self, data_dir, save_dir, side, augmentation_apply, input_channel="L"):
        """              
        """
        self.data_dir = Path(data_dir) # Z:\Projects\Angiogram\Data\Processed\Zijun\UpdateTrainingPipline\data\label&image
        self.save_dir = Path(save_dir)
        self.input_channel = input_channel
        self.augmentation = augmentation_apply
        self.side = side
        self.image_path = glob.glob(os.path.join(self.data_dir, "image", "*.png"))
        self.infor_pd = read_csv(self.data_dir /'patient.csv')

        if side == "ALL":
            self.indices = self.infor_pd.index_one.tolist()
        elif side == "R":
            self.indices = self.infor_pd.index_one[self.infor_pd['side']=='R'].tolist()
        elif side == "L":
            self.indices = self.infor_pd.index_one[self.infor_pd['side']=='L'].tolist()

        print(f"Indices on {side} side: {','.join([str(i) for i in self.indices])}")

    def __getitem__(self, index):
        # print(f"Try to get index {index}")
        # match_index = self.indices[index]
        image_raw = Image.open(self.data_dir/'image'/f'{index}-frame.png').convert("L")
        image_seg = Image.open(self.data_dir/'label'/f'{index}-seg.png').convert("L")
        transformed_raw, transformed_seg = transform_image(image_raw, image_seg, self.input_channel, self.augmentation)

        return index, transformed_raw, transformed_seg

    def __len__(self):
        return len(self.indices)

    def get_save_path(self, folder):
        # Used in testing image saving for Trainer
        return self.save_dir / folder 
    
    def get_target_index(self):
        # Used in Data Loader Construction
        return self.indices

class CombinedDataset(D.ConcatDataset):
    """
    Dataset as a concatenation of different datasets.
    dirpath(Path object from pathlib): the dirctory where all the datasets are stored
    dataset_names_by_user(string or list of strings): provided by user, folder names that could found under the dirpath
    dataset_class_handles(dictionary[dataset_folder_name] = Class_handle): defined in data_loaders.py
    input_channel(str): either 'RGB' or 'L'
    augmentation_apply(dictionary[input_channel] = a_tranformation_list): defined in data_loaders.py
    obtain_label(bool): wether or not to obtain the labels in these datasets
    training(bool): are these dataset used for training

    """
    def __init__(self, dirpath, dataset_names_by_user, dataset_class_handles, input_channel, 
        augmentation_apply, obtain_label, training):
        
        # Obtain the path of the data dir and the dataset folder names under this dir 
        self.dirpath = Path(dirpath)
        dataset_folder_names = os.listdir(self.dirpath)

        # initate the dataset_list and len_dataset_list to store the 
        # different dataset instance and their length
        dataset_list, self.len_dataset_list = [], []
        # iterate through the dataset_names_by_user provided by user
        for dataset_name in dataset_names_by_user:
            # make sure this dataset exist in the folder and is defined in the class handle dictionary
            if dataset_name in dataset_folder_names and dataset_name in dataset_class_handles:           
                dataset_full_path = self.dirpath / dataset_name
                # create an instance of the dataset class 
                dataset_created = dataset_class_handles[dataset_name](dataset_full_path, 
                    augmentation_apply, input_channel, obtain_label, training)
                dataset_list.append(dataset_created)
                self.len_dataset_list.append(len(dataset_created))
            else:
                raise KeyError("Based on the dataset folder name: {} provided, \
                the folder either doesn't exist, or not defined.".format(dataset_name))
        
        # create a new ConcatDataset instance
        super().__init__(dataset_list)

        # obtain the labels of each dataset and concatnate them together
        self.label = []
        for each_dataset in dataset_list:
            label_each_dataset = each_dataset.get_label()
            if label_each_dataset is not None:
                self.label += label_each_dataset

    def get_label(self):
        return self.label

    def get_dataset_length(self):
        return self.len_dataset_list


class Public_Dataset(D.Dataset):

    def __init__(self, dirpath, augmentation_apply, input_channel, obtain_label, training):

        self.dirpath = Path(dirpath)
        self.raw_dirpath = self.dirpath / 'raw'
        self.segmented_dirpath = self.dirpath /'annotated'

        self.input_channel = input_channel
        self.training = training
        self.obtain_label = obtain_label
        self.augmentation = augmentation_apply
        self.patient_info = read_csv(self.dirpath / 'patient.csv', names=["image","label"])

    def get_label(self):
        if self.obtain_label is True:
            return list(self.patient_info["label"])
        else:
            return None

    def __getitem__(self, index):
        # add this line of index_ just in case the image(name) row is somehow shuffled 
        index_ = self.patient_info.loc[index,"image"]
        image_raw = Image.open(self.raw_dirpath / "{}.png".format(index_)).convert("L")
        image_seg = Image.open(self.segmented_dirpath / "{}.png".format(index_)).convert("L")

        transformed_raw, transformed_seg = transform_image(image_raw, image_seg, self.input_channel, self.augmentation)

        return index, transformed_raw, transformed_seg, -1

    def __len__(self):
        return len(fnmatch.filter(os.listdir(self.raw_dirpath), "*.png"))

class Alberto_Dataset(D.Dataset):
    """
    This is a private dataset from alberto lab that contains 343 segmented images,
    The quality of the annotaion is low. 
    """
    def __init__(self, dirpath, augmentation_apply, input_channel, obtain_label, training):

        self.dirpath = Path(dirpath)
        self.raw_dirpath = self.dirpath / 'images'
        self.segmented_dirpath = self.dirpath /'annotations'
        self.image_names = fnmatch.filter(os.listdir(self.raw_dirpath), "*.png")
        # Read into the patient info
        self.patient_info = read_csv(self.dirpath / 'patient.csv', names=["image","label"])

        # Set the image name column as index, so it's easier to match with the labels
        self.patient_info.set_index("image", inplace=True)

        self.input_channel = input_channel
        self.training = training
        self.obtain_label = obtain_label
        self.augmentation = augmentation_apply

    def get_label(self):
        if self.obtain_label is True:
            # get the raw image names
            raw_image_names = [image_name.replace('.png','') for image_name in self.image_names]
            
            return [self.patient_info.loc[raw_image_name]["label"] for raw_image_name in raw_image_names]
        else:
            return None

    def __getitem__(self, index):

        # Get the image name with given index
        image_name = self.image_names[index]
        # image_name_segmented = image_names_segmented[image_names_segmented.index(image_name_raw)]

        # Read Images, Transform and Augment Images
        image_raw = Image.open(self.raw_dirpath / image_name)#.convert("L")
        image_seg = Image.open(self.segmented_dirpath / image_name)#.convert("L")
        transformed_raw, transformed_seg = transform_image(image_raw, image_seg, self.input_channel, self.augmentation)
        return index, transformed_raw, transformed_seg, -1

    def __len__(self):
    
        return len(fnmatch.filter(os.listdir(self.raw_dirpath), "*.png"))

def normalize_image(image_raw, set_min_val=0, set_max_val=1):
    unsqueeze_image_raw = np.expand_dims(np.array(image_raw), axis=0)
    # np.array(image_raw).shape = (512,512) -> return shape (512,512)
    # np.array(image_raw).shape = (1,512,512) -> return shape (512,512)
    # np.array(image_raw).shape = (2,512,512) -> return shape (2,512,512)
    outputs = []
    for idx, _input in enumerate(unsqueeze_image_raw):
        # Get from 'class RangeNormalize()'
        # https://github.com/ncullen93/torchsample/blob/master/torchsample/transforms/tensor_transforms.py
        _min_val = _input.min()
        _max_val = _input.max()
        a = (set_max_val - set_min_val) / (_max_val - _min_val)
        b = set_max_val - a * _max_val
        _input = _input * a + b
        outputs.append(_input)
    normalized_unsqueeze_image_raw = outputs if idx > 1 else outputs[0]
    if len(normalized_unsqueeze_image_raw)==1:
        normalized_image_raw = np.squeeze(normalized_unsqueeze_image_raw, axis=0)
        return normalized_image_raw
    else:
        return normalized_unsqueeze_image_raw

def transform_image(image_raw, image_segmentation, self_input_channel, transformation_applied):
    
    transformation_applied = transformation_applied[self_input_channel]
    
    if self_input_channel == 'RGB':
        image_raw = image_raw.convert('RGB')
        image_segmentation = image_segmentation.convert('RGB')
    else: # apply 2DminmaxNormalization when the input_channel is 'L'
        image_raw = normalize_image(image_raw)

    image_segmentation = np.array(image_segmentation)
    image_segmentation = np.array(image_segmentation>(image_segmentation.max()/2),dtype=np.uint8)

    # assert len(np.unique(np.array(image_segmentation)))==2

    if transformation_applied is not None:
        segmap = SegmentationMapsOnImage(np.array(image_segmentation), shape=np.array(image_segmentation).shape)
        image_raw_aug, image_segmentation_aug = transformation_applied(image=np.array(image_raw), segmentation_maps=segmap)
        image_segmentation_aug = np.array(image_segmentation_aug.get_arr())
    else:
        image_raw_aug = np.array(image_raw)
        image_segmentation_aug = np.array(image_segmentation)

    image_raw_aug = torch.from_numpy(image_raw_aug)
    image_segmentation_aug = torch.from_numpy(image_segmentation_aug)
    image_raw_aug = image_raw_aug.float()
    image_segmentation_aug = image_segmentation_aug.float()
    # assert len(np.unique(np.array(image_segmentation_aug)))==2

    # Modify the transformed images and correct their dimensions
    if self_input_channel == 'RGB':
        # if the image size is NxN, then for RGB case (three channel input)
        # image_raw_aug.shape = (3,512,512) and image_segmentation_aug.shape = (3,512,512)
        # but request input and output is:
        # image_raw_aug.shape = (3,512,512) and image_segmentation_aug.shape = (1,512,512)
        image_segmentation_aug = image_segmentation_aug[0,:,:] # (3,512,512) -> (512,512)
        image_segmentation_aug = image_segmentation_aug.unsqueeze(0)
    else:
        # if the image size is NxN, then for L case (one channel input)
        # image_raw_aug.shape = (512,512) and image_segmentation_aug.shape = (512,512)
        # but request input and output is:
        # image_raw_aug.shape = (1,512,512) and image_segmentation_aug.shape = (1,512,512)
        image_raw_aug = image_raw_aug.unsqueeze(0)
        image_segmentation_aug = image_segmentation_aug.unsqueeze(0)

    # assert len(np.unique(np.array(image_segmentation_aug)))==2

    return image_raw_aug, image_segmentation_aug

