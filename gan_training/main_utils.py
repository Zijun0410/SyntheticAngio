import torch
import torch.nn as nn
import numpy as np
import os, re, sys, math

def weights_init(m):
    # Custom weights initialization given by the DCGAN paper
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def check_loss(loss, loss_name):
    if not math.isfinite(loss.item()):
        print(f"{loss_name} Losses are {loss}, stopping training")
        sys.exit(1)

def calculate_gradient_penalty(model, real_images, synthetic_images, device):
    """
    Calculates the gradient penalty loss for WGAN GP
        https://github.com/Lornatang/WassersteinGAN_GP-PyTorch
    """
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * synthetic_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_save_name_from_hyper_params(hyper_params): 
    """
    Inputs:
        hyper_params <python dict>: all the hyper-params as a dictionary
    Output:
        <python string>: "{hyper-param-name}_{hyper-param-value}"
    """
    output_list = []
    for key, value in list(hyper_params.items()):
        output_list.append(f"{key}_{value}")
    name = '_'.join(output_list)

    return name

def get_the_latest_ckpt(chkpt_folder):
    """
    Inputs:
        chkpt_folder <python string>: the path to the folder containing all the checkpoints
    Output:
        <python string>: the path to the latest checkpoint
    """
    epoch_num_list, path_list = [], []
    for file in os.listdir(chkpt_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".pt") and 'ckpt' in filename: 
            epoch = re.findall(r'\d+', filename)
            if len(epoch) > 0 :
                epoch_num_list.append(int(epoch[0]))
                path_list.append(filename)
    if len(epoch_num_list) == 0:
        return None
    max_epoch = max(epoch_num_list)
    return path_list[epoch_num_list.index(max_epoch)]
