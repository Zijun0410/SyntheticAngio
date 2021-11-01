import torch
import numpy as np
from skimage.io import (imread, imsave)
from pathlib import Path

import segmentation_models_pytorch as smp
import platform
import sys


class UNet_SMP(smp.Unet):
    """
    A change of backbone U-net model build with the following API
    API: https://github.com/qubvel/segmentation_models.pytorch
    Installation: pip install segmentation-models-pytorch
    """
    def __init__(self, backbone='resnet101', in_channels=1, attention=None):
        self.encoder = backbone
        super().__init__(encoder_name=self.encoder, in_channels=in_channels, 
            encoder_weights='imagenet', activation='sigmoid', 
            decoder_attention_type=attention) 

def otsu(input_vector):
    """
    The value of input_vector range from 0 to 1
    """
    input_vector = input_vector*255
    pixel_number = len(input_vector)
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(input_vector, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: 
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        if pcb == 0 or pcf == 0:
            continue
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight
        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)        
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    final_thresh = final_thresh/255
    return final_thresh


def apply_2DminmaxNormalization(image_raw, set_min_val=0, set_max_val=1):
    """
    Helper function of dlm_classifier
    apply 2D min/max Normalization on raw images.
    input: d*n*n numpy array
    output: d*n*n numpy array
    """
    outputs = np.zeros(image_raw.shape)
    for idx, _input in enumerate(image_raw):
        # Get from 'class RangeNormalize()'
        # https://github.com/ncullen93/torchsample/blob/master/torchsample/transforms/tensor_transforms.py
        _min_val = _input.min()
        _max_val = _input.max()
        a = (set_max_val - set_min_val) / (_max_val - _min_val)
        b = set_max_val - a * _max_val
        _input = _input * a + b
        outputs[idx] = _input
    return outputs

def load_image(database_path, frame_count):
    """
    Helper funtion of dlm_classifier
    Load image given the database_path and frame_count
    image_stack is a numpy array of size B*512*512 with B=frame_count
    """
    image_stack = np.zeros((frame_count,512,512))
    for frame_number in range(frame_count):
        img = imread(database_path / f"frame{frame_number+1}.png", as_gray=True)
        img = np.array(img)
        image_stack[frame_number] = np.reshape(img, (512, 512))
    return image_stack

def save_image(database_path, frame_count, output):
    """
    Helper funtion of dlm_classifier
    Save image given the database_path and frame_count
    """
    for frame_number in range(len(output)):
        pred = output[frame_number][0] # shape = (512,512)
        pred = pred.detach().numpy() # Change tensor into numpy array
        vec_ostu_threshold = otsu(np.reshape(pred, -1)) # Get the Ostu's threshould
        pred_binary = np.array(pred>vec_ostu_threshold,dtype=np.uint8)*255 # Binarize the image
        imsave(database_path / f"dlm{frame_number+1}.png", pred_binary) # Save image

def generate_model_path():

    if platform.system()=='Linux':
        default_model_dir = Path(r'/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/ensemble_model')
    elif platform.system()=='Windows':
        default_model_dir = Path(r'Z:\Projects\Angiogram\Data\Processed\ensemble_model') 
    else:
        raise EOFError("Platform not defined")

    return default_model_dir

def densenet_classifier(database_path, frame_count):
    """
    A classifier based on the deep learning model 
    frame_count is the total number of frames selected from one video

    """
    # Load saved model 
    model_path = generate_model_path()
    frame_count = int(frame_count)
    device = torch.device('cpu')
    state_dict = torch.load(model_path / "densenet_unet_model_best_dict.pth", map_location=device)
    # state_dict = checkpoint['state_dict']
    model = UNet_SMP(backbone = "densenet121", in_channels = 1)
    model.load_state_dict(state_dict)
    model = model.to(device)
    # Model Eval would compromise the performance, I have not investigate why.
    # model.eval()

    # Load the image and make transformation
    database_path = Path(database_path)
    image_stack = load_image(database_path, frame_count)
    image_transformed = apply_2DminmaxNormalization(image_stack)
    with torch.no_grad():
        # Turn the array into a tensor
        data = torch.as_tensor(image_transformed, dtype=torch.float32, device=device)
        # Model receive an input of size [B,1,512,512] where B is the batch size equaling to 
        # frame_count. image_transformed is of size [B,512,512] and we need to unsqueeze the tensor.
        data = torch.unsqueeze(data, 1)

        # Feed data into the model 
        output = model(data)

        # Save the images
        save_image(database_path, frame_count, output)

# if __name__ == '__main__':
#     database_path = Path('/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/results/updatedPipeline_0609/UKL/Full/001 (3)_L')
#     frame_count=3
#     densenet_classifier(database_path, frame_count)


if __name__ == "__main__":
    a = '%r' % str(sys.argv[1])
    b = '%r' % str(sys.argv[2])
    img_path = a[1:-1]
    frame_count = int(b[1:-1])
    densenet_classifier(img_path, frame_count)