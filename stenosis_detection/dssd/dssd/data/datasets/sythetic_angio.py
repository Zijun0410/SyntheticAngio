import os
import torch.utils.data as torchData
import numpy as np
from PIL import Image
import pandas as pd
import glob
from dssd.structures.container import Container

class SytheticDataset(torchData.Dataset):
    def __init__(self, data_dir, batch_number, transform=None, target_transform=None):

        # data_dir: <pathlib Path object>, the base dir where all the data were loaded
        #          Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output\')
        # bacth_number: <python int>, the batch number 
        self.box_radi = 10
        self.ref_size = 512
        self.data_dir = data_dir / f'{bacth_number}'
        self.transform = transform
        self.target_transform = target_transform
        self.video_list = [Path(dir_path).stem for dir_path in glob.glob(os.path.join(self.data_dir, "*", ""))]

    def __getitem__(self, index):
        video_name = self.video_list[index]
        # load the image as a PIL Image
        image = self._read_image(self.data_dir / video_name)

        # load the labels and bounding boxes in x1, y1, x2, y2 order.
        boxes, labels = _get_annotation(self.data_dir / video_name)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels
        )
        # return the image, the targets and the index in your dataset
        return image, targets, index

    def _read_image(self, infor_dir):
        image_file = infor_dir / "sythetic.png" 
        image = Image.open(image_file).convert("L")
        image = np.array(image)
        return image

    def _get_annotation(self, infor_dir):
        stenosis_df = pd.read_csv(infor_dir / "stenosis_infor.csv", )
        boxes = [self._cal_box(x, y) for x, y in zip(stenosis_df['x_center'], stenosis_df['y_center'])]
        lables = stenosis_df['degree']
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _cal_box(self, x_center, y_center):
        x_min = np.floor(x_center - self.box_radi)
        x_max = np.floor(x_center + self.box_radi)
        y_min = np.floor(y_center - self.box_radi)
        y_max = np.floor(y_center + self.box_radi)
        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        return {"height": self.ref_size, "width": self.ref_size}

    def get_annotation(self, index):
        video_name = self.video_list[index]
        return video_name, self._get_annotation(self.data_dir / video_name)

