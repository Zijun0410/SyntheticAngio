import torch.utils.data as D
from pathlib import Path, PureWindowsPath
import pandas as pd
import cv2
import numpy as np

# Define the root path
def return_root_path():
    """ Return the root path for data saving given the current OS """
    if Path.cwd().as_posix().startswith('/'):
        return Path('/nfs/turbo/med-kayvan-lab/')
    return Path('Z:/')

# Obtain all the paths to the images from the csv file
def get_image_infor(dir_list, file_name):
    for i, dir_ in enumerate(dir_list):
        dir_ = return_root_path() / PureWindowsPath(dir_.name)
        image_infor = pd.read_csv(dir_ / file_name)
        if i == 0:
            df_combined = image_infor
        else:
            df_combined = df_combined.append(image_infor)
    return df_combined

class ImageBlend(D.Dataset):
    """
    Dataset for poisson image blending
        the real bakcground image, 
        the synthetic volumn,
        and the synthetic segmentation.
    Input:
        dir_list <list of python string>: include the directory of the images
        file_name <python string>: the name of the csv file that contains the information of the image
        kernel_size <python int>: the size of the kernel for dialating the segmentation image
    """
    def __init__(self, dir_list, kernel_size, file_name='stenosis_detail.csv'):
        super(ImageBlend, self).__init__()
        self.root_path = return_root_path()
        self.image_infor = get_image_infor(dir_list, file_name)
        self.folder_col_id = self.image_infor.columns.get_loc('output_folder')
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def adjust_dir(self, str_dir):
        image_dir = Path(str_dir.replace('+', '/'))
        return self.root_path / image_dir

    def __getitem__(self, index):
        str_dir = self.adjust_dir(self.image_infor.iloc[index, self.folder_col_id])    
        source_img = cv2.imread(str(str_dir/'volumn.png'), cv2.IMREAD_GRAYSCALE)
        segmentation = cv2.imread(str(str_dir/'segmentation.png'), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.dilate(segmentation, self.kernel, iterations=1)
        target_img = cv2.imread(str(str_dir/'background.png'), cv2.IMREAD_GRAYSCALE)
        # Return the source, mask and target image as well as the image_loading directory
        return source_img, mask_img, target_img, str_dir

    def __len__(self):
        return len(self.image_infor)