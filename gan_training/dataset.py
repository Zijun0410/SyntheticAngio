import torch.utils.data as D
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

from data_utils import basic_transformations

# Define the root path
def return_root_path():
    """ Return the root path for data saving given the current OS """
    if Path.cwd().as_posix().startswith('/'):
        return Path('/nfs/turbo/med-kayvan-lab/')
    return Path('Z:/')

# Obtain all the paths to the images from the csv file
def get_image_infor(dir_list, file_name):
    for i, dir_ in enumerate(dir_list):
        image_infor = pd.read_csv(Path(dir_) / file_name)
        if i == 0:
            df_combined = image_infor
        else:
            df_combined = df_combined.append(image_infor)
    return df_combined

# Map the name of the transformation to the function
transformation_dict = dict(zip(['None', 'basic'], [None, basic_transformations]))

# Open the image with path and turn it into a tensor
open_image_as_tensor = lambda x: T.PILToTensor()(Image.open(x).convert("L"))

class RealImage(D.Dataset):
    """
    Dataset for Real X-ray Angiograhy Images
    Input:
        dir_list <list of python string>: include the directory of the image
        file_name <python string>: the name of the csv file that contains the information of the image
        transform <python string>: the name of the transformation
    """
    # umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UMR\Full')
    # ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UKR\Full')
    # dir_list = [umr_dir, ukr_dir]

    def __init__(self, dir_list, transform='None', file_name='image_infor.csv'):
        super(RealImage, self).__init__()
        self.image_infor = get_image_infor(dir_list, file_name)
        self.root_path = return_root_path()
        self.transform = transformation_dict[transform]
    
    def adjust_dir(self, str_dir):
        image_dir = Path(str_dir.replace('+', '/'))
        return self.root_path / image_dir.relative_to("Z:/")

    def __getitem__(self, index):
        str_dir = self.adjust_dir(self.image_infor.iloc[index, 1])
        image = open_image_as_tensor(str_dir)
        if self.transform:
            image = self.transform(image)
        return index, image

    def __len__(self):
        return len(self.image_infor)


class GeneratorInput(D.Dataset):
    """
    Dataset for Generator Input, containing 
        the real bakcground image, 
        the synthetic volumn,
        and the synthetic mask.
    Input:
        dir_list <list of python string>: include the directory of the images
        file_name <python string>: the name of the csv file that contains the information of the image
        transform <python string>: the name of the transformation
    """
    def __init__(self, dir_list, transform='None', file_name='stenosis_detail.csv'):
        super(GeneratorInput, self).__init__()
        self.image_infor = get_image_infor(dir_list, file_name)
        self.root_path = return_root_path()
        self.transform = transformation_dict[transform]
        self.folder_col_id = self.image_infor.columns.get_loc('output_folder')

    def adjust_dir(self, str_dir):
        image_dir = Path(str_dir.replace('+', '/'))
        return self.root_path / image_dir

    def __getitem__(self, index):
        str_dir = self.adjust_dir(self.image_infor.iloc[index, self.folder_col_id])    
        image_volumn = open_image_as_tensor(str_dir/'volumn.png')
        image_mask = open_image_as_tensor(str_dir/'mask.png')
        image_background = open_image_as_tensor(str_dir/'background.png')
        images = torch.cat([image_volumn, image_mask, image_background], dim=0)
        if self.transform:
            images = self.transform(images)
        return index, images

    def __len__(self):
        return len(self.image_infor)


if __name__ == "__main__":
    umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UMR\Full')
    ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UKR\Full')
    dir_list = [umr_dir, ukr_dir]
    real_image = RealImage(dir_list)
    print(len(real_image))
    print(real_image[1].shape)

    umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UoMR')
    ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UKR')
    dir_list = [umr_dir, ukr_dir]
    generator_input = GeneratorInput(dir_list)
    print(len(generator_input))
    print(generator_input[1].shape)
