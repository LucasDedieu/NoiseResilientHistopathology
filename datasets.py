import os
import torch
from extraction import Extractor
import custom_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import cv2


DATASETS = ["crc", "pcam", "bach", "mhist", "lc", "gashis"]
BACKBONES = ["ctranspath", "dino", "phikon", "retccl", "pathoduet", "bt", "swav", "moco", "ibot"]

CRC_CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
CRC_IMAGES_PATH = '/home/primaa/Lucas/crc/'

PCAM_CLASS_NAMES = ['benign','metastasis']
PCAM_IMAGES_PATH = '/home/primaa/Lucas/pcam/from_PCAM/'

MHIST_CLASS_NAMES = ['HP','SSA']
MHIST_IMAGES_PATH = '/home/primaa/Lucas/mhist/'

BACH_CLASS_NAMES = ['benign', 'insitu', 'invasive', 'normal']
BACH_IMAGES_PATH = '/media/primaa/nfs_nas/external_databases/from_BACH/'

LC_CLASS_NAMES = ['lung_benign', 'lung_aca', 'lung_scc', 'colon_benign', 'colon_aca']
LC_IMAGES_PATH = '/home/primaa/Lucas/lc/'


GASHIS_CLASS_NAMES= ['abnormal', 'normal']
GASHIS_IMAGES_PATH = "D:/NoisyLabels/dataset/gashis/"

DF_PATH = "./deep_features"


def get_dataset(device, config, logger, noise_rate=0, seed=42):
    """
    Retrieves and prepares the dataset based on the provided configuration.

    Args:
        device (torch.device): The device (CPU or GPU) to use for data processing.
        config (object): Configuration object containing dataset settings.
        logger (logging.Logger): Logger instance for logging messages.
        noise_rate (float, optional): Proportion of noisy labels to introduce. Default is 0.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing class names, training data (X_train, y_train),
               validation data (X_val, y_val), and test data (X_test, y_test).

    Raises:
        SystemExit: If the dataset name is not supported.
    """
    dataset_name = config.dataset.name
    if dataset_name not in DATASETS:
        logger.error("Wrong dataset name. Only accept %s",DATASETS)
        sys.exit(1)
    logger.info("Dataset: %s", dataset_name)

    class_names = globals()[dataset_name.upper()+"_CLASS_NAMES"]
    if config.dataset.type == "images":
        X_train, y_train, X_val, y_val, X_test, y_test = get_dataset_images(config, logger, noise_rate=noise_rate, seed=seed)
    elif config.dataset.type == "deep_features":
        X_train, y_train, X_val, y_val, X_test, y_test = get_dataset_df(device, config, logger, noise_rate=noise_rate, seed=seed)
    
    X_train, y_train = add_noise(logger, X_train, y_train, noise_rate, seed=seed)
    logger.info("Train: %s", np.unique(y_train, return_counts=True))
    logger.info("Test: %s", np.unique(y_test, return_counts=True))
    
    return class_names, X_train, y_train, X_val, y_val, X_test, y_test



def get_dataset_images(config, logger, noise_rate, seed):
    """
    Loads and prepares image dataset.

    Args:
        config (object): Configuration object containing dataset settings.
        logger (logging.Logger): Logger instance for logging messages.
        noise_rate (float): Proportion of noisy labels to introduce.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing training data (X_train, y_train),
               validation data (X_val, y_val), and test data (X_test, y_test).
    """
    dataset_name = config.dataset.name
    path = globals()[dataset_name.upper()+"_IMAGES_PATH"]
    train_dataset = ImageFolder(root=path+"train/")
    X_train = [train_dataset.imgs[i][0] for i in range(len(train_dataset))]
    y_train = [train_dataset.targets[i] for i in range(len(train_dataset))]

    try:
        test_dataset = ImageFolder(root=path+"test/")
        X_test = [test_dataset.imgs[i][0] for i in range(len(test_dataset))]
        y_test = [test_dataset.targets[i] for i in range(len(test_dataset))]
    except FileNotFoundError:
        logger.warn("No test dataset founded. Splitting train into train/test")
        test_size = int(0.2*len(train_dataset))
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, 
            train_size=len(train_dataset)-test_size, 
            test_size=test_size, 
            stratify=y_train, 
            random_state=seed
        )
        
    try:
        val_dataset = ImageFolder(root=path+"val/")
        X_val = [val_dataset.imgs[i][0] for i in range(len(val_dataset))]
        y_val = [val_dataset.targets[i] for i in range(len(val_dataset))]
    except FileNotFoundError:
        logger.warn("No validation dataset founded. Splitting train into train/val")
        val_size = int(config.dataset.val_size * len(X_train))
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            train_size=len(X_train)-val_size, 
            test_size=val_size, 
            stratify=y_train, 
            random_state=seed
        )
        
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test



def get_dataset_df(device, config, logger, noise_rate, seed):
    """
    Loads and prepares deep feature dataset.

    Args:
        device (torch.device): The device (CPU or GPU) to use for data processing.
        config (object): Configuration object containing dataset settings.
        logger (logging.Logger): Logger instance for logging messages.
        noise_rate (float): Proportion of noisy labels to introduce.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing training data (X_train, y_train),
               validation data (X_val, y_val), and test data (X_test, y_test).

    Raises:
        SystemExit: If the backbone name is not supported.
    """
    dataset_name = config.dataset.name
    backbone_name = config.dataset.backbone

    if backbone_name not in BACKBONES:
        logger.error("Wrong backbone name. Only accept %s",BACKBONES)
        sys.exit(1)
    logger.info("Backbone: %s",backbone_name)

    path = os.path.join(DF_PATH, dataset_name)

    if not already_extracted(logger, path, backbone_name):
        logger.info("DF dataset not found. Procedding extraction...")
        X_train, y_train, X_val, y_val, X_test, y_test = get_dataset_images(config, logger, noise_rate=0, seed=seed)
        train_set = ImgDataset(X_train, y_train, transform=custom_transforms.val_images())
        val_set = ImgDataset(X_val, y_val, transform=custom_transforms.val_images())
        test_set = ImgDataset(X_test, y_test, transform=custom_transforms.val_images())
        train_loader = DataLoader(train_set, batch_size=16)#, num_workers=config.dataset.num_workers)
        val_loader = DataLoader(val_set, batch_size=16)
        test_loader = DataLoader(test_set, batch_size=16)
        extractor = Extractor(device=device, config=config, logger=logger)
        extractor.extract(train_loader, val_loader, test_loader)
        logger.info("Extraction done")
    
    X_train = np.load(os.path.join(path,"train_df_"+backbone_name+".npy"))
    y_train = np.load(os.path.join(path,"train_labels_"+backbone_name+".npy"))
    X_val = np.load(os.path.join(path,"val_df_"+backbone_name+".npy"))
    y_val = np.load(os.path.join(path,"val_labels_"+backbone_name+".npy"))
    X_test = np.load(os.path.join(path,"test_df_"+backbone_name+".npy"))
    y_test = np.load(os.path.join(path,"test_labels_"+backbone_name+".npy"))

    return X_train, y_train, X_val, y_val, X_test, y_test



def already_extracted(logger, path, backbone_name):
    """
    Checks if the deep features have already been extracted.

    Args:
        logger (logging.Logger): Logger instance for logging messages.
        path (str): Path to the directory containing the extracted features.
        backbone_name (str): Name of the backbone used for feature extraction.

    Returns:
        bool: True if all required files are found, False otherwise.
    """
    try:
        files = os.listdir(path)
        if "train_df_"+backbone_name+".npy" not in files:
            return False
        if "train_labels_"+backbone_name+".npy" not in files:
            return False
        if "val_df_"+backbone_name+".npy" not in files:
            return False
        if "val_labels_"+backbone_name+".npy" not in files:
            return False
        if "test_df_"+backbone_name+".npy" not in files:
            return False
        if "test_labels_"+backbone_name+".npy" not in files:
            return False
        return True
    except FileNotFoundError:
        return False



def add_noise(logger, X, y, noise_rate, seed=42):
    """
    Introduces noise into the labels of the dataset.

    Args:
        logger (logging.Logger): Logger instance for logging messages.
        X (numpy.ndarray): Feature data.
        y (numpy.ndarray): Label data.
        noise_rate (float): Proportion of noisy labels to introduce.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing the feature data (X) and the noisy labels (y_noisy).
    """
    if noise_rate <= 0:
        logger.info("Noise rate must be positive or null. Returning original labels")
        return X, y
    np.random.seed(seed)

    num_classes = len(np.unique(y))
    num_samples = len(y)
    
    num_noisy_samples = int(noise_rate * num_samples)

    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)

    y_noisy = np.copy(y)
    for idx in noisy_indices:
        current_class = y[idx]
        # Ensure the new label is different from the current label
        new_class = (current_class + 1 + np.random.randint(num_classes - 1)) % num_classes
        y_noisy[idx] = new_class

    actual_noise = (y_noisy != y).mean()
    logger.info("Actual noise: %s",actual_noise)
    return X, y_noisy



class ImgDataset(Dataset):
    """
    Image dataset

    Args:
        image_paths (np.array): Numpy array of image file paths.
        labels (np.array): Numpy array of labels corresponding to the images.
        transform (callable, optional): Optional transform to apply to the images. Defaults to None.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        
        label = self.labels[idx]
        return image, label



class DFDataset(Dataset):
    """
    Deep feature dataset.

    Args:
        deep features (np.array): Numpy array of images.
        labels (np.array): Numpy array of labels corresponding to the images.
    """
    def __init__(self, deep_features, labels, transform=None):
        self.deep_features = deep_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.deep_features)

    def __getitem__(self, idx):
        deep_feature = self.deep_features[idx].astype(np.float64)        
        if self.transform:
            deep_feature = self.transform(deep_feature)
        deep_feature = torch.from_numpy(deep_feature).type(torch.float32)
        label = self.labels[idx]
        return deep_feature, label