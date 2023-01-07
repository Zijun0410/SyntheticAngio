# Packages
import numpy as np
import torch
import torch.optim as optim
from skimage.io import imsave
from utils import compute_gt_gradient, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix
from data_load import ImageBlend
from pathlib import Path
import torchvision.transforms as transforms

transform = transforms.ToTensor()

# Define LBFGS optimizer directly on the input image
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def main():
    # Initialize the ImageBlend class for image loading 
    kernel_size = 3
    umr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UoMR')
    ukr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UKR')
    dir_list = [umr_dir, ukr_dir]
    imageBlend = ImageBlend(dir_list, kernel_size)

    # Intialize the device for GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import VGG network for computing style and content loss
    mean_shift = MeanShift(device)
    vgg = Vgg16().to(device)

    # Default weights for loss functions
    grad_weight = 1e4; style_weight = 1e4; content_weight = 1; tv_weight = 1e-6
    num_steps = 1000

    for idx in range(len(imageBlend)):
        # Get the source, mask, and target images as well as the save directory
        source_img, mask_img, target_img, save_dir = imageBlend[idx]
        mask_img[mask_img>0] = 1

        # Compute Ground-Truth Gradients
        # obtain the source and target image tensors
        # torch.Size([1, 3, 512, 512]), torch.Size([1, 3, 512, 512], torch.Size([3, 512, 512]
        source_img, target_img, gt_gradient = compute_gt_gradient(source_img, target_img, mask_img, device)
        # Intilize the input image tensor of torch.Size([1, 3 512, 512])
        input_img = torch.randn(source_img.shape).to(device)

        # Make image mask tensor of torch.Size([1, 3, 512, 512])
        mask_img = transform(mask_img).unsqueeze(0).repeat(1, 3, 1, 1).float().to(device)

        # Initialize the optimizer 
        optimizer = get_input_optimizer(input_img)

        # Define Loss Functions
        mse = torch.nn.MSELoss()
        
        ###################################
        ########### First Pass ###########
        ###################################
        run = [0]
        while run[0] <= num_steps:
            def closure():
                # Composite Foreground and Background to Make Blended Image
                blend_img = torch.zeros(target_img.shape).to(device)
                blend_img = input_img*mask_img + target_img*(mask_img-1)*(-1) 
                
                # Compute Laplacian Gradient of Blended Image
                pred_gradient = laplacian_filter_tensor(blend_img, device) # torch.Size([3, 512, 512])
                
                # Compute Gradient Loss
                grad_loss = 0
                for c in range(len(pred_gradient)):
                    grad_loss += mse(pred_gradient[c], gt_gradient[c])
                grad_loss /= len(pred_gradient)
                grad_loss *= grad_weight

                # Compute Style Loss
                target_features_style = vgg(mean_shift(target_img))
                target_gram_style = [gram_matrix(y) for y in target_features_style]
                
                blend_features_style = vgg(mean_shift(input_img))
                blend_gram_style = [gram_matrix(y) for y in blend_features_style]
                
                style_loss = 0
                for layer in range(len(blend_gram_style)):
                    style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
                style_loss /= len(blend_gram_style)  
                style_loss *= style_weight           

                # Compute Content Loss
                blend_obj = blend_img
                source_object_features = vgg(mean_shift(source_img*mask_img))
                blend_object_features = vgg(mean_shift(blend_obj*mask_img))
                content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
                content_loss *= content_weight
                
                # Compute TV Reg Loss
                tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                        torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
                tv_loss *= tv_weight
                
                # Compute Total Loss and Update Image
                loss = grad_loss + style_loss + content_loss + tv_loss
                optimizer.zero_grad()
                loss.backward()
                
                # Print Loss
                if run[0] % 10 == 0:
                    print("run {}:".format(run))
                    print('grad : {:4f}, style : {:4f}, content: {:4f}, tv: {:4f}'.format(\
                                grad_loss.item(), \
                                style_loss.item(), \
                                content_loss.item(), \
                                tv_loss.item()
                                ))
                    print()
                
                run[0] += 1
                return loss

            optimizer.step(closure)

        # clamp the pixels range into 0 ~ 255
        input_img.data.clamp_(0, 255)

        # Make the Final Blended Image
        blend_img = torch.zeros(target_img.shape).to(device)
        blend_img = input_img*mask_img + target_img*(mask_img-1)*(-1) 
        blend_img_np = blend_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

        # Save image from the first pass
        imsave(save_dir /'deep_blend.png', blend_img_np.astype(np.uint8))

        break

if __name__ == '__main__':
    main()


# ###################################
# ########### Second Pass ###########
# ###################################

# # Default weights for loss functions in the second pass
# style_weight = 1e7; content_weight = 1; tv_weight = 1e-6
# ss = 512; ts = 512
# num_steps = opt.num_steps

# first_pass_img_file = 'results/'+str(name)+'_first_pass.png'
# first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))
# target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
# first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(device)
# target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(device)

# first_pass_img = first_pass_img.contiguous()
# target_img = target_img.contiguous()

# # Define LBFGS optimizer
# def get_input_optimizer(first_pass_img):
#     optimizer = optim.LBFGS([first_pass_img.requires_grad_()])
#     return optimizer

# optimizer = get_input_optimizer(first_pass_img)

# print('Optimizing...')
# run = [0]
# while run[0] <= num_steps:
    
#     def closure():
        
#         # Compute Loss Loss    
#         target_features_style = vgg(mean_shift(target_img))
#         target_gram_style = [gram_matrix(y) for y in target_features_style]
#         blend_features_style = vgg(mean_shift(first_pass_img))
#         blend_gram_style = [gram_matrix(y) for y in blend_features_style]
#         style_loss = 0
#         for layer in range(len(blend_gram_style)):
#             style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
#         style_loss /= len(blend_gram_style)  
#         style_loss *= style_weight        
        
#         # Compute Content Loss
#         content_features = vgg(mean_shift(first_pass_img))
#         content_loss = content_weight * mse(blend_features_style.relu2_2, content_features.relu2_2)
        
#         # Compute Total Loss and Update Image
#         loss = style_loss + content_loss
#         optimizer.zero_grad()
#         loss.backward()
        
#         # Print Loss
#         if run[0] % 1 == 0:
#             print("run {}:".format(run))
#             print(' style : {:4f}, content: {:4f}'.format(\
#                           style_loss.item(), \
#                           content_loss.item()
#                           ))
#             print()
        
#         run[0] += 1
#         return loss
    
#     optimizer.step(closure)

# # clamp the pixels range into 0 ~ 255
# first_pass_img.data.clamp_(0, 255)

# # Make the Final Blended Image
# input_img_np = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

# # Save image from the second pass
# imsave('results/'+str(name)+'_second_pass.png', input_img_np.astype(np.uint8))



