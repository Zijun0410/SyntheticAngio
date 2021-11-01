import imgaug.augmenters as iaa

nature_paper_augmentation = {}
# Deep learning segmentation of major vessels in X-ray coronary angiography
# " Data augmentation was performed with rotation (−20° to 20°), 
#   translation shift (0–10% of image size in horizontal and vertical axes), 
#   and zoom (0–10%).""
nature_paper_augmentation['L'] = iaa.Sequential([
    iaa.Sometimes(0.7, iaa.Affine(scale=(1, 1.1), rotate=(-20, 20), translate_percent=(-0.1,0.1)))
    ], random_order=False)

nature_paper_augmentation['RGB'] = iaa.Sequential([
    iaa.Sometimes(0.7, iaa.Affine(scale=(1, 1.1), rotate=(-20, 20), translate_percent=(-0.1,0.1)))
    ], random_order=False)

custom_augmentation = {}

custom_augmentation['L'] = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.Affine(scale=(1, 1.1), rotate=(-10, 10), translate_percent=(-0.1,0.1)))
    ], random_order=False)

custom_augmentation['RGB'] = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.Affine(scale=(1, 1.1), rotate=(-10, 10), translate_percent=(-0.1,0.1)))
    ], random_order=False)

# custom_augmentation = {}
# custom_augmentation['RGB'] = [
#     albu.Resize(512, 512),
#     albu.ShiftScaleRotate(shift_limit=(0,0.1), scale_limit=(0,0.1), rotate_limit=(-20,20), p = 0.7),
#     albu.ElasticTransform(),
#     ToTensor()]
# custom_augmentation['L'] = [
#     albu.Resize(512, 512),
#     albu.ShiftScaleRotate(shift_limit=(0,0.1), scale_limit=(0,0.1), rotate_limit=(-20,20), p = 0.7),
#     albu.ElasticTransform(),
#     ToTensor()]

no_augmentation = {}
no_augmentation['RGB'] = None
no_augmentation['L'] = None