from torchvision import transforms

basic_transformations = transforms.Compose(
    [
        transforms.PILToTensor(),
        # transforms.Lambda(crop_my_image),
    ]
)