from load_data import ImageBlend
import poisson_blend as poisson
from pathlib import Path
import numpy as np
import cv2

def main():
    kernel_size = 3
    umr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UoMR')
    ukr_dir=Path(r'Projects\Angiogram\Data\Processed\Zijun\Synthetic\GAN_Data\UKR')
    dir_list = [umr_dir, ukr_dir]

    imageBlend = ImageBlend(dir_list, kernel_size)

    for idx in range(len(imageBlend)):
        # Get the source, mask, and target images as well as the save directory
        source_img, mask_img, target_img, save_dir = imageBlend[idx]
        # Normalize mask to range [0,1]
        mask = np.atleast_3d(mask_img).astype(np.float) / 255.
        # Make mask binary
        mask[mask != 1] = 0
        # Trim to one channel
        mask = mask[:,:,0]
        source_img = source_img[:,:,0]
        target_img = target_img[:,:,0]
        # Call the poisson method on each individual channel
        output = poisson.process(source_img, target_img, mask)
        # Merge the channels back into one image
        # result = cv2.merge(output)
        # normal_clone = cv2.seamlessClone(source_img, target_img, mask_img, (256,256), cv2.NORMAL_CLONE)
        # cv2.illuminationChange(source_img, target_img, mask_img)
        # cv.MONOCHROME_TRANSFER
        # normal_clone = cv2.seamlessClone(source_img, target_img, mask_img, (256,256), cv2.MONOCHROME_TRANSFER)
        cv2.imwrite(str(save_dir / f'naiv_poisson_blend_k_{kernel_size}.png'), output)
        break
        
if __name__ == '__main__':
    main()