'''
datasets.py
'''

# ============================================== IMPORTS ============================================== #

from torch.utils.data import Dataset
import h5py
import json
import numpy as np
import random
import utils

import pdb

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    # def __init__(self, task, train_band_means, train_band_stds):
    def __init__(self, task):
        self.task = task
        # self.train_band_means = np.array(train_band_means)
        # self.train_band_stds = np.array(train_band_stds)

        with h5py.File(f'{data_dir_path}/{task}/{task}_h5.hdf5', 'r') as h5_file:
            self.tile_count = len(h5_file['Sentinel2'])
            self.sentinel2 = h5_file['Sentinel2'][:]
            self.task_data = h5_file[task][:]
            self.tile_ids = h5_file['id'][:]

            print(f'{task} tile count: {self.tile_count}')
            print(f'Sentinel-2: {self.sentinel2.shape}')
            print(f'{task}: {self.task_data.shape}')

        self._get_split_data()

    def _get_split_data(self):
        random.seed(42)
        random.shuffle(self.tile_ids)
        end_train_ids = int(0.7 * self.tile_count)
        end_val_ids = int(0.85 * self.tile_count)

        train_ids = self.tile_ids[:end_train_ids].tolist()
        val_ids = self.tile_ids[end_train_ids:end_val_ids].tolist()
        test_ids = self.tile_ids[end_val_ids:].tolist()
        print(f'Dataset lengths: Train = {len(train_ids)}, Val = {len(val_ids)}, Test = {len(test_ids)}')

        train_indices = np.array([np.where(self.tile_ids == tile_id)[0][0] for tile_id in train_ids])
        train_images = self.sentinel2[train_indices]
        self.train_band_means = train_images.mean(axis=(0,2,3))[:, None, None].tolist()
        self.train_band_stds = train_images.std(axis=(0,2,3))[:, None, None].tolist()
        split_data = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids, 'train_band_means': self.train_band_means, 'train_band_stds': self.train_band_stds}

        with open('split_data.json', 'w') as file:
            json.dump(split_data, file, indent=4)

    def __len__(self):
        return self.tile_count

    def __getitem__(self, tile_id):
        index = np.where(self.tile_ids == tile_id)[0][0]
        sentinel2 = self.sentinel2[index]
        sentinel2 = (sentinel2 - self.train_band_means) / self.train_band_stds # normalization
        task_data = self.task_data[index]

        return sentinel2, task_data
