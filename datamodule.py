'''
datamodule.py
'''

# ============================================== IMPORTS ============================================== #

from dataset import MMEarthBenchDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import utils

# ============================================== CLASSES ============================================== #

class DataModule(LightningDataModule):
    def __init__(self, task, split_type, batch_size, data_dir_path, num_workers, seed):
        super().__init__()

        self.task = task
        self.split_type = split_type
        self.batch_size = batch_size
        self.data_dir_path = data_dir_path
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = MMEarthBenchDataset(task=task, data_dir_path=data_dir_path)

        self.setup('init')

    def setup(self, stage):
        # self.dataset = MMEarthBenchDataset(task=self.task, data_dir_path=self.data_dir_path)
        # split_data = self.dataset.split_data

        # for split in ['train', 'val', 'random_test', 'geographic_test']:
        #     setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=split_data[f'{split}_indices']))

        if stage == 'init':
            splits = ['train']
        if stage == 'fit':
            splits = ['val']
        elif stage == 'test':
            splits = ['random_test', 'geographic_test']

        for split in splits:
            setattr(self, f'{split}_dataset', Subset(dataset=self.dataset, indices=self.dataset.split_data[f'{split}_indices']))

        if self.task != 'species' and stage == 'test':
            self._calculate_test_rmse_with_train_mean()

        # self.train_dataset = Subset(dataset=self.dataset, indices=split_data['train_indices']) # TODO: MODIFY TRAIN INDICES TO GET A SUBSET
        # self.val_dataset = Subset(dataset=self.dataset, indices=split_data['val_indices'])
        # self.random_test_dataset = Subset(dataset=self.dataset, indices=split_data['random_test_indices'])
        # self.geographic_test_dataset = Subset(dataset=self.dataset, indices=split_data['geographic_test_indices'])

        # if self.task != 'species' and stage == 'fit':
        #     self._calculate_test_rmse_with_train_mean()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        random_test_dataloader = DataLoader(self.random_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        geographic_test_dataloader = DataLoader(self.geographic_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        return [random_test_dataloader, geographic_test_dataloader]

    def _calculate_test_rmse_with_train_mean(self):
        train_values = np.array([task_value.squeeze() for _, task_value in self.train_dataset]).ravel()

        if self.task == 'biomass':
            train_values = [value for value in train_values if value != -9999]

        train_mean = np.mean(train_values)

        print(f'Mean of train values: {round(float(train_mean), 2)}')

        val_values = np.array([task_value.squeeze() for _, task_value in self.val_dataset]).ravel()

        if self.task == 'biomass':
            val_values = [value for value in val_values if value != -9999]

        val_rmse = np.sqrt(np.mean((np.array(val_values) - train_mean) ** 2))

        print(f'Val RMSE using the train mean as the prediction: {round(float(val_rmse), 2)}')

        random_test_values = np.array([task_value.squeeze() for _, task_value in self.random_test_dataset]).ravel()

        if self.task == 'biomass':
            random_test_values = [value for value in random_test_values if value != -9999]

        random_test_rmse = np.sqrt(np.mean((np.array(random_test_values) - train_mean) ** 2))

        print(f'Random test RMSE using the train mean as the prediction: {round(float(random_test_rmse), 2)}')

        geographic_test_values = np.array([task_value.squeeze() for _, task_value in self.geographic_test_dataset]).ravel()

        if self.task == 'biomass':
            geographic_test_values = [value for value in geographic_test_values if value != -9999]

        geographic_test_rmse = np.sqrt(np.mean((np.array(geographic_test_values) - train_mean) ** 2))

        print(f'Geographic test RMSE using the train mean as the prediction: {round(float(geographic_test_rmse), 2)}')
