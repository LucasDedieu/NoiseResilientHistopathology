import models
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from models import get_backbone
import util

DF_PATH = "deep_features"


class Extractor(nn.Module):
    """
    Extractor class for extracting deep features from images using a specified backbone model.

    Args:
        device (torch.device): The device (CPU or GPU) to use for data processing.
        config (mlconfig.Config): Configuration object containing dataset settings.
        logger (logging.Logger): Logger instance for logging messages.
    """

    def __init__(self, device, config, logger):
        super(Extractor, self).__init__()
        self.config = config
        self.logger = logger
        self.backbone_name = config.dataset.backbone
        self.backbone = get_backbone(logger, self.backbone_name)
        self.device = device



    def extract(self,train_loader, val_loader, test_loader):
        """
        Extracts deep features from the provided data loaders and saves them to disk.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        """
        backbone = self.backbone.to(self.device)

        backbone.eval()
        with torch.no_grad():
            
            df_train = []
            labels_train = []
            for batch in tqdm(train_loader):
                images, labels = batch
                images = images.to(self.device)
                if self.backbone_name == "pathoduet":
                    df = backbone.forward_features(images)
                else:
                    df = backbone(images)
                if self.backbone_name == "phikon":
                    df= df.last_hidden_state[:, 0, :]
                df = df.cpu().numpy().tolist()
                df_train = df_train+df
                labels_train = labels_train + labels.tolist()
            df_train = np.array(df_train)
            labels_train = np.array(labels_train)

            df_val = []
            labels_val = []
            for batch in tqdm(val_loader):
                images, labels = batch
                images = images.to(self.device)
                if self.backbone_name == "pathoduet":
                    df = backbone.forward_features(images)
                else:
                    df = backbone(images)
                if self.backbone_name == "phikon":
                    df= df.last_hidden_state[:, 0, :]
                df = df.cpu().numpy().tolist()
                df_val = df_val+df
                labels_val = labels_val + labels.tolist()
            df_val = np.array(df_val)
            labels_val = np.array(labels_val)

            df_test = []
            labels_test = []
            for batch in tqdm(test_loader):
                images, labels = batch
                images = images.to(self.device)
                if self.backbone_name == "pathoduet":
                    df = backbone.forward_features(images)
                else:
                    df = backbone(images)
                if self.backbone_name == "phikon":
                    df= df.last_hidden_state[:, 0, :]
                df = df.cpu().numpy().tolist()
                df_test = df_test+df
                labels_test = labels_test + labels.tolist()
            df_test = np.array(df_test)
            labels_test = np.array(labels_test)

        self.write_numpy(df_train, labels_train, df_val, labels_val, df_test, labels_test)



    def write_numpy(self, df_train, labels_train, df_val, labels_val, df_test, labels_test):
        """
        Saves the extracted features and labels to disk as .npy files.

        Args:
            df_train (numpy.ndarray): Extracted features from the training data.
            labels_train (numpy.ndarray): Labels corresponding to the training data features.
            df_val (numpy.ndarray): Extracted features from the validation data.
            labels_val (numpy.ndarray): Labels corresponding to the validation data features.
            df_test (numpy.ndarray): Extracted features from the test data.
            labels_test (numpy.ndarray): Labels corresponding to the test data features.
        """
        dataset_name = self.config.dataset.name
        path = os.path.join(DF_PATH, dataset_name)
        util.build_dirs(path)

        np.save(os.path.join(path, "train_df_"+self.backbone_name), df_train)
        np.save(os.path.join(path, "train_labels_"+self.backbone_name), labels_train)

        np.save(os.path.join(path, "val_df_"+self.backbone_name), df_val)
        np.save(os.path.join(path, "val_labels_"+self.backbone_name), labels_val)

        np.save(os.path.join(path, "test_df_"+self.backbone_name), df_test)
        np.save(os.path.join(path, "test_labels_"+self.backbone_name), labels_test)
