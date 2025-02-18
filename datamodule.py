'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
import torch
import utils

# ============================================== CLASSES ============================================== #

class DataModule(LightningDataModule):
    # def __init__(self, task, dataset_class, split_data, batch_size, num_workers):
    def __init__(self, task, dataset_class, batch_size, num_workers):
        super().__init__()

        self.task = task
        self.dataset_class = dataset_class
        # self.split_data = split_data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        # self.dataset = self.dataset_class(task=self.task, train_band_means=self.split_data['train_band_means'], train_band_stds=self.split_data['train_band_stds'])
        self.dataset = self.dataset_class(task=self.task)
        self.split_data = utils.read_json('split_data.json')

        if stage == 'fit':
            self.train_dataset = Subset(dataset=self.dataset, indices=self.split_data['train_ids'])
            self.val_dataset = Subset(self.dataset, self.split_data['val_ids'])
        elif stage == 'validate':
            self.val_dataset = Subset(self.dataset, self.split_data['val_ids'])
        elif stage == 'test':
            self.test_dataset = Subset(self.dataset, self.split_data['test_ids'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
