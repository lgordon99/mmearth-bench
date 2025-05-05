'''
dataset.py
'''

# ============================================== IMPORTS ============================================== #

from torch.utils.data import Dataset
import h5py
import json
import numpy as np
import torch
import utils

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, data_dir_path):
        self.task = task
        self.data_dir_path = data_dir_path

        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            self.sentinel2 = h5_file['Sentinel2'][:]
            self.task_data = h5_file[task][:]
            self.tile_count = len(self.sentinel2) # number of tiles for the task

            print(f'{task} tile count: {self.tile_count}')
            print(f'Sentinel-2: {self.sentinel2.shape}')
            print(f'{task}: {self.task_data.shape}')

        self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        # FIX
        if task == 'species':
            train_species_counts = task_data[train_indices].sum(axis=0)
            species_with_too_few_train_observations = np.where(train_species_counts < 50)[0]
            task_data[np.ix_(test_indices, species_with_too_few_train_observations)] = 0

    def __len__(self):
        return self.tile_count

    def __getitem__(self, index):
        sentinel2 = self.sentinel2[index]
        sentinel2 = (sentinel2 - self.split_data['train_band_means']) / self.split_data['train_band_stds'] # normalization
        task_data = self.task_data[index]

        if len(task_data.shape) == 2: # for biomass
            task_data = np.expand_dims(task_data, axis=0)

        return torch.tensor(sentinel2, dtype=torch.float32), torch.tensor(task_data, dtype=torch.float32)
