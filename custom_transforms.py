import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def train_images(normalize=True, resize=224):
    """
    Define augmentation pipeline for train images.

    Args:s
        normalize (bool, optional): Whether to apply normalization. Defaults to True.
        resize (int, optional): Size to resize the images. Defaults to 224.

    Returns:
        albumentations.Compose: Composed transformations for training images.
    """
    transform_list = []

    transform_list.extend([
        A.Resize(width=resize, height=resize, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.5, hue=(-0.15, 0.15), saturation=(0.8, 1.2), brightness=(0.7, 1.2), contrast=(0.7, 1.5)),
    ])

    if normalize:
        transform_list.append(A.Normalize(mean=mean, std=std))
    else:
        transform_list.append(A.Normalize(mean=[.0,.0,.0], std=[1.,1.,1.]))

    transform_list.append(ToTensorV2())
    return A.Compose(transform_list)



def val_images(normalize=True, resize=224):
    """
    Define augmentation pipeline for validation and test images.

    Args:
        normalize (bool, optional): Whether to apply normalization. Defaults to True.
        resize (int, optional): Size to resize the images. Defaults to 224.

    Returns:
        albumentations.Compose: Composed transformations for validation images.
    """
    transform_list = []
    #transform_list.append(A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT))
    transform_list.append(A.Resize(width=resize, height=resize, interpolation=cv2.INTER_LINEAR))
    if normalize:
        transform_list.append(A.Normalize(mean=mean, std=std))
    else:
        transform_list.append(A.Normalize(mean=[.0,.0,.0], std=[1.,1.,1.]))

    transform_list.append(ToTensorV2())
    return A.Compose(transform_list)



def train_df(sigma=0.1):
    """
    Define augmentation pipeline for train deep features.

    Args:
        sigma (float, optional): Maximum standard deviation for Gaussian blur. Defaults to 0.1.

    Returns:
        GaussianBlur: Instance of GaussianBlur transformation.
    """
    return GaussianBlur(sigma=sigma)



class GaussianBlur():
    """
    Apply Gaussian blur transformation with stardard deviation between 0 and sigma.

    Args:
        sigma (float, optional): Maximum standard deviation for Gaussian blur. Defaults to 0.1.
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, input):
        s = np.random.uniform(0,self.sigma)
        return input+np.random.normal(size=input.shape, scale=s)