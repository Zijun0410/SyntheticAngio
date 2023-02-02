
import torch.utils.data as D
from pathlib import Path, PureWindowsPath
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

from data_utils import basic_transformations
import cv2
import numpy as np

STENOSIS_CLASSES = {0: 'na', 1: 'stenosis', 2: 'severe'}
COLORS = np.random.uniform(0, 255, size=(len(STENOSIS_CLASSES), 3))

# Define the root path
def return_root_path():
    """ Return the root path for data saving given the current OS """
    if Path.cwd().as_posix().startswith('/'):
        return Path('/nfs/turbo/med-kayvan-lab/')
    return Path('Z:/')

# Obtain all the paths to the images from the csv file
def get_image_infor(dir_list, file_name):
    for i, dir_ in enumerate(dir_list):
        dir_ = return_root_path() / PureWindowsPath(dir_).relative_to("Z:/")
        image_infor = pd.read_csv(dir_ / file_name)
        if i == 0:
            df_combined = image_infor
        else:
            df_combined = df_combined.append(image_infor)
    # reset the index
    df_combined = df_combined.reset_index(drop=True)
    return df_combined

# Map the name of the transformation to the function
transformation_dict = dict(zip(['None', 'basic'], [None, basic_transformations]))

# Open the image with path and turn it into a tensor
def open_image_as_tensor(x):
    img = transforms.PILToTensor()(Image.open(x).convert("RGB"))
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    return img

class SyntheticImage(D.Dataset):
    """
    Dataset for Synthetic X-ray Angiograhy Images
    Input:
        dir_list <list of python string>: include the directory of the image
        file_name <python string>: the name of the csv file that contains the information of the stensosis
        transform <python string>: the name of the transformation
    """
    # umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\UoMR')
    # ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\UKR')
    # dir_list = [umr_dir, ukr_dir]

    def __init__(self, dir_list, transform='None', file_name='stenosis_detail.csv'):
        self.root_path = return_root_path()
        self.image_infor = get_image_infor(dir_list, file_name)
        self.transform = transformation_dict[transform]
    
    def adjust_dir(self, str_dir):
        image_dir = Path(str_dir.replace('+', '/'))
        return self.root_path / image_dir / 'synthetic_1.png'

    def __getitem__(self, index):
        str_dir = self.adjust_dir(self.image_infor.loc[index, 'output_folder'])
        x_center = self.image_infor.loc[index, 'x_center']
        y_center = self.image_infor.loc[index, 'y_center']
        degree = torch.tensor(self.image_infor.loc[index, 'degree'])
        effect_region = self.image_infor.loc[index, 'effect_region']
        diameter = int((16+round(effect_region*100))/2)
        box_tensor = torch.tensor([x_center-diameter, y_center-diameter, x_center+diameter, y_center+diameter]).unsqueeze(0).float()
        degree = 1* (degree > 0)

        image = open_image_as_tensor(str_dir)
        if self.transform:
            image = self.transform(image)

        target = {}
        target['boxes'] = box_tensor
        target['labels'] = degree.unsqueeze(0)

        return target, image.float()

    def __len__(self):
        return len(self.image_infor)


class RealImage(D.Dataset):
    """
    Dataset for Real X-ray Angiograhy Images
    Input:
        dir_list <list of python string>: include the directory of the image
        file_name <python string>: the name of the csv file that contains the information of the stensosis
        transform <python string>: the name of the transformation
    """
    # umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UMR\Full')
    # ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Real_Image\UKR\Full')
    # dir_list = [umr_dir, ukr_dir]

    def __init__(self, dir_list, transform='None', file_name='image_infor.csv', name='UMR'):
        super(RealImage, self).__init__()
        self.root_path = return_root_path()
        self.image_infor = get_image_infor(dir_list, file_name)
        self.transform = transformation_dict[transform]
        self.name = name
    
    def adjust_dir(self, str_dir):
        image_dir = Path(str_dir.replace('+', '/'))
        return self.root_path / image_dir.relative_to("Z:/")

    def __getitem__(self, index):
        str_dir = self.adjust_dir(self.image_infor.iloc[index, 1])
        image = open_image_as_tensor(str_dir)
        if self.transform:
            image = self.transform(image)
        return index, image.float()

    def save_image(self, index, boxes, labels, output_dir):
        # Get all the predicited class names.
        classes = [STENOSIS_CLASSES[i] for i in labels]
        str_dir = self.adjust_dir(self.image_infor.iloc[index, 1])
        image = cv2.imread(str(str_dir))
        updated_image = self.draw_boxes(boxes, classes, labels, image)
        save_dir = output_dir / self.name 
        name_str = str_dir.name.split('.')[0]
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f'{str_dir.parent.name}_{name_str}.png'), updated_image)
        
    def draw_boxes(self, boxes, classes, labels, image):
        """
        Draws the bounding box around a detected object.
        """
        lw = max(round(sum(image.shape) / 2 * 0.003), 2) # Line width.
        tf = max(lw - 1, 1) # Font thickness.
        
        for i, box in enumerate(boxes):
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            color = COLORS[labels[i]]
            class_name = classes[i]
            cv2.rectangle(
                image,
                p1,
                p2,
                color[::-1],
                thickness=lw,
                lineType=cv2.LINE_AA
            )
            # For filled rectangle.
            w, h = cv2.getTextSize(
                class_name, 
                0, 
                fontScale=lw / 3, 
                thickness=tf
            )[0]  # Text width, height
            
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            
            cv2.rectangle(
                image, 
                p1, 
                p2, 
                color=color[::-1], 
                thickness=-1, 
                lineType=cv2.LINE_AA
            )  
            # cv2.putText(
            #     image, 
            #     class_name, 
            #     (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            #     cv2.FONT_HERSHEY_SIMPLEX, 
            #     fontScale=lw / 3.8, 
            #     color=(255, 255, 255), 
            #     thickness=tf, 
            #     lineType=cv2.LINE_AA
            # )
        return image

    def __len__(self):
        return len(self.image_infor)
    
if __name__ == "__main__":
    umr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\UoMR')
    ukr_dir=Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\UKR')
    dir_list = [umr_dir, ukr_dir]
    real_image = SyntheticImage(dir_list)
    print(len(real_image))