'''
datasets.py
'''

# ============================================== IMPORTS ============================================== #

from boundingbox import BoundingBox
from rasterio.enums import Resampling
from rtree.index import Index, Property
from torch.utils.data import Dataset
from torchgeo.datasets import concat_samples
import h5py
import numpy as np
import rasterio
import torch
import utils

import pdb

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
# nodata_value = utils.read_yaml('config.yml')['nodata_value']

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task):
        self.task = task

        with h5py.File(f'{data_dir_path}/{task}/{task}_h5.hdf5', 'r') as h5_file:
            self.tile_count = len(h5_file['Sentinel2'])
            print(f'{task} tile count: {self.tile_count}')
            print(h5_file.keys())

            self.sentinel2 = h5_file['Sentinel2']
            print(f'Sentinel-2: {self.sentinel2.shape}')

            self.task_data = h5_file[task]
            print(f'{task}: {self.task_data.shape}')

            self.tile_ids = h5_file['id'][()]

    def __len__(self):
        return self.tile_count

    def __getitem__(self, tile_id):
        index = np.where(self.tile_ids == tile_id)[0][0]
        sentinel2 = self.sentinel2[index]
        task_data = self.task_data[index]
        print(sentinel2.shape)
        print(task_data.shape)

        return sentinel2, task_data

mmearth_bench_dataset = MMEarthBenchDataset(task='soil_nitrogen')
