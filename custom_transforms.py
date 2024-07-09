import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



def train_images(normalize=True, resize=224):
    """
    Creates a transformation pipeline for training images.

    Args:
        normalize (bool, optional): Whether to normalize the images. Default is True.
        resize (int, optional): The size to which images should be resized. Default is 224.

    Returns:
        albumentations.core.composition.Compose: A composed transformation pipeline.
    """
    transform_list = []

    transform_list.extend([
        A.Resize(width=resize, height=resize, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.5, hue=(-0.15, 0.15), saturation=(0.8, 1.2), brightness=(0.7, 1.2), contrast=(0.7, 1.5)),
    ])

    if normalize:
        transform_list.append(A.Normalize(mean=MEAN, std=STD))

    transform_list.append(ToTensorV2())
    return A.Compose(transform_list)



def val_images(normalize=True, resize=224):
    """
    Creates a transformation pipeline for validation images.

    Args:
        normalize (bool, optional): Whether to normalize the images. Default is True.
        resize (int, optional): The size to which images should be resized. Default is 224.

    Returns:
        albumentations.core.composition.Compose: A composed transformation pipeline.
    """
    transform_list = []
    #transform_list.append(A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT))
    transform_list.append(A.Resize(width=resize, height=resize, interpolation=cv2.INTER_LINEAR))
    if normalize:
        transform_list.append(A.Normalize(mean=MEAN, std=STD))

    transform_list.append(ToTensorV2())
    return A.Compose(transform_list)



def train_df(sigma=0.1):
    """
    Creates a Gaussian blur transformation for deep features.

    Args:
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 0.1.

    Returns:
        GaussianBlur: An instance of the GaussianBlur transformation.
    """
    return GaussianBlur(sigma=sigma)



class GaussianBlur():
    """
    Gaussian blur transformation for deep features.

    Args:
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 0.1.
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, input):
        s = np.random.uniform(0,self.sigma)
        return input+np.random.normal(size=input.shape, scale=s)

