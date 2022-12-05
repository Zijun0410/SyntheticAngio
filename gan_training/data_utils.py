from torchvision import transforms

basic_transformations = transforms.Compose(
    [
        transforms.PILToTensor(),
        # transforms.Lambda(crop_my_image),
    ]
)

# import random

# def my_segmentation_transforms(image, segmentation):
#     if random.random() > 0.5:
#         angle = random.randint(-30, 30)
#         image = transforms.functional.rotate(image, angle)
#         segmentation = transforms.functional.rotate(segmentation, angle)
#     # more transforms ...
#     return image, segmentation

# def crop_my_image(image: PIL.Image.Image) -> PIL.Image.Image:
#     """Crop the images so only a specific region of interest is shown to my PyTorch model"""
#     left, right, width, height = 20, 80, 40, 60
#     return transforms.functional.crop(image, left=left, top=top, width=width, height=height)