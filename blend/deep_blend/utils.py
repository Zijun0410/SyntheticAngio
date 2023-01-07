# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:28:28 2019

@author: Owen and Tarmily
"""

from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch
from torchvision import models
from collections import namedtuple
import time
import torchvision.transforms as transforms
import asyncio

transform = transforms.ToTensor()

def laplacian_filter_tensor(img_tensor, device):

    laplacian_filter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    # The weight must be a tensor with shape (out_channels, in_channels/groups, kH, kW)
    laplacian_conv.weight = nn.Parameter(torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(device))
    
    for param in laplacian_conv.parameters():
        param.requires_grad = False
    if img_tensor.shape[1] == 1:
        gradient_tensor = laplacian_conv(img_tensor).squeeze(1)  # torch.Size([3, 512, 512])
    elif img_tensor.shape[1] == 3:
        gradient_tensor = [laplacian_conv(img_tensor[:, i, :, :].unsqueeze(1)).squeeze(1) for i in range(3)] # torch.Size([3, 512, 512])

    return gradient_tensor
    

def compute_gt_gradient(source_img, target_img, mask, device):
    """
    compute gradient of source image and target image. source_img, target_img and mask all have a shape of (512, 512)
    """
    # compute source image gradient
    source_img_tensor = transform(source_img).unsqueeze(0).float().to(device) # torch.Size([1, 1, 512, 512])
    source_gradient_tenosr = laplacian_filter_tensor(source_img_tensor, device) # torch.Size([3, 512, 512]) 
    source_gradient_numpy = source_gradient_tenosr[0].cpu().data.numpy()[0] # (512, 512)
   
    # compute target image gradient
    target_img_tensor = transform(target_img).unsqueeze(0).float().to(device) # torch.Size([1, 1, 512, 512])
    target_gradient_tenosr = laplacian_filter_tensor(target_img_tensor, device) # torch.Size([3, 512, 512])
    target_gradient_numpy = target_gradient_tenosr[0].cpu().data.numpy()[0] # (512, 512)
    
    # foreground gradient
    foreground_gradient = source_gradient_numpy * mask
   
    # background gradient
    background_gradient = target_gradient_numpy * (mask - 1) * (-1)
 
    # add up foreground and background gradient
    gt_gradient = foreground_gradient + background_gradient
    gt_gradient_tensor = torch.from_numpy(gt_gradient).unsqueeze(0).float().to(device)

    # torch.Size([1, 3, 512, 512]), torch.Size([1, 3, 512, 512], torch.Size([3, 512, 512]
    return source_img_tensor.repeat(1, 3, 1, 1), target_img_tensor.repeat(1, 3, 1, 1), gt_gradient_tensor.repeat(3, 1, 1)


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std



class MeanShift(nn.Conv2d):
    def __init__(self, device):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range=1
        rgb_mean=(0.4488, 0.4371, 0.4040)
        rgb_std=(1.0, 1.0, 1.0)
        sign=-1
        std = torch.Tensor(rgb_std).to(device)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(device) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(device) / std
        for p in self.parameters():
            p.requires_grad = False


def get_matched_features_numpy(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False)
    cpu_blended_features = blended_features.cpu().detach().numpy()
    cpu_target_features = target_features.cpu().detach().numpy()
    for filter in range(0, blended_features.size(1)):
        matched_filter = torch.from_numpy(hist_match_numpy(cpu_blended_features[0, filter, :, :],
                                                           cpu_target_features[0, filter, :, :])).to(blended_features.device)
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def get_matched_features_pytorch(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False).to(blended_features.device)
    for filter in range(0, blended_features.size(1)):
        matched_filter = hist_match_pytorch(blended_features[0, filter, :, :], target_features[0, filter, :, :])
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def hist_match_pytorch(source, template):

    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    return matched_source.reshape(oldshape)


async def hist_match_pytorch_async(source, template, index, storage):

    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        storage[0, index, :, :] = source.reshape(oldshape)
        return

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    storage[0, index, :, :] = matched_source.reshape(oldshape)


async def loop_features_pytorch(source, target, storage):
    size = source.shape
    tasks = []

    for i in range(0, size[1]):
        task = asyncio.ensure_future(hist_match_pytorch_async(source[0, i], target[0, i], i, storage))
        tasks.append(task)

    await asyncio.gather(*tasks)


def get_matched_features_pytorch_async(source, target, matched):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(loop_features_pytorch(source, target, matched))
    loop.run_until_complete(future)
    loop.close()


def hist_match_numpy(source, template):

    oldshape = source.shape

    source = source.ravel()
    template = template.ravel()

    max_val = max(source.max(), template.max())
    min_val = min(source.min(), template.min())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    source_hist, source_bin_edges = np.histogram(a=source, bins=num_bins, range=(min_val, max_val))
    template_hist, template_bin_edges = np.histogram(a=template, bins=num_bins, range=(min_val, max_val))

    hist_bin_centers = source_bin_edges[:-1] + hist_step / 2.0

    source_quantiles = np.cumsum(source_hist).astype(np.float32)
    source_quantiles /= source_quantiles[-1]
    template_quantiles = np.cumsum(template_hist).astype(np.float32)
    template_quantiles /= template_quantiles[-1]

    index_function = np.vectorize(pyfunc=lambda x: np.argmin(np.abs(template_quantiles - x)))

    nearest_indices = index_function(source_quantiles)

    source_data_bin_index = np.clip(a=np.round(source / hist_step), a_min=0, a_max=num_bins-1).astype(np.int32)

    mapped_indices = np.take(nearest_indices, source_data_bin_index)
    matched_source = np.take(hist_bin_centers, mapped_indices)

    return matched_source.reshape(oldshape)


def main():
    size = (64, 512, 512)
    source = np.random.randint(low=0, high=500000, size=size).astype(np.float32)
    target = np.random.randint(low=0, high=500000, size=size).astype(np.float32)
    source_tensor = torch.Tensor(source).to(0)
    target_tensor = torch.Tensor(target).to(0)
    matched_numpy = np.zeros(shape=size)
    matched_pytorch = torch.zeros(size=size, device=0)

    numpy_time = time.process_time()

    for i in range(0, size[0]):
        matched_numpy[i, :, :] = hist_match_numpy(source[i], target[i])
    
    numpy_time = time.process_time() - numpy_time

    pytorch_time = time.process_time()

    for i in range(0, size[0]):
        matched_pytorch[i, :, :] = hist_match_pytorch(source_tensor[i], target_tensor[i])
    
    pytorch_time = time.process_time() - pytorch_time


if __name__ == "__main__":
    main()
