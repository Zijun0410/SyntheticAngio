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
    def __init__(self, data_dir, augmentation_apply, salient_frame_number,input_channel="L",):
        """              
        """
        self.dirpath = Path(data_dir) 
        self.input_channel = input_channel
        self.augmentation = augmentation_apply
        self.subdirpath = glob.glob(os.path.join(self.dirpath,"*",""))
        self.sfn = salient_frame_number # salient_frame_number
        print(f"The Frame Number is {self.sfn}")

    def __getitem__(self, index):
        image_raw_dirpath = Path(self.subdirpath[index//self.sfn]) /'frame{}.png'.format(index%self.sfn +1)
        image_raw = Image.open(image_raw_dirpath).convert("L")

        transformed_raw, _ = transform_image(image_raw, image_raw, self.input_channel, self.augmentation)
        return index, transformed_raw

    def __len__(self):
        return len(self.subdirpath)*self.sfn

    def get_save_path(self, index):
        return Path(self.subdirpath[index//self.sfn]) /'dlm{}.png'.format(index%self.sfn +1)
    
    def get_target_index(self):
        return list(range(len(self.subdirpath)*self.sfn))


        
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

