import numpy as np
from skimage import color, io
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from utils import *


class places365DataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=32, train_prop=0.8):
        super().__init__()
        self.train_path = path
        self.test_path = path
        self.batch_size = batch_size
        self.train_prop = train_prop

        self.train_transform = transforms.Compose(
            [transforms.RandomCrop(224,pad_if_needed=True),
             transforms.RandomHorizontalFlip(),
             PILToNumpyRGB, 
             RGBToLAB, 
             NormalizeValues,
             transforms.ToTensor()]
             )
        self.test_transform = transforms.Compose(
            [transforms.Resize(224),
             PILToNumpyRGB, 
             RGBToLAB, 
             NormalizeValues,
             transforms.ToTensor()]
             )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.full_dataset = torchvision.datasets.ImageFolder(root=self.train_path, transform=self.train_transform)
            train_size = int(len(self.full_dataset) * self.train_prop)
            val_size = len(self.full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.ImageFolder(root=self.test_path, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           pin_memory=True)
