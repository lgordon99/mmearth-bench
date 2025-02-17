'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch

import pdb

# ============================================== CLASSES ============================================== #

class DataModule(LightningDataModule):
    def __init__(self, task, dataset_class, batch_size, num_workers):
        super().__init__()

        self.task = task
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset = self.dataset_class(task=self.task)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset=self.dataset, lengths=[0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
