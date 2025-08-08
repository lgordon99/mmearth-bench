'''
dataset.py
'''

# ============================================== IMPORTS ============================================== #

from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import utils

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, data_dir_path):
        self.task = task
        self.data_dir_path = data_dir_path
        self.no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')

        if task == 'pretraining':
            h5_path = '/n/gajos_lab/Lab/luciagordon/MMEarth-train/data_1M_v001_64/data_1M_v001_64.h5'

            with h5py.File(h5_path, 'r') as h5_file:
                print(h5_file.keys())
                self.sentinel2 = h5_file['sentinel2'][:]
                print(f'Sentinel-2: {self.sentinel2.shape}')
        else:
            h5_path = f'{data_dir_path}/{task}/{task}.h5'
            self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        with h5py.File(h5_path, 'r') as h5_file:
            self.modality_data = {modality: h5_file[modality][:] for modality in self.no_data_values.keys()}

            if task != 'pretraining':
                self.task_data = h5_file[task][:]

        # # convert categorical modalities to one-hot encoding
        # self.modality_data['DynamicWorld'] = np.eye(self.no_data_values['DynamicWorld']+1)[self.modality_data['DynamicWorld'].astype(int).squeeze(1)].transpose(0, 3, 1, 2)
        # self.modality_data['ESA_WorldCover'] = np.eye(self.no_data_values['ESA_WorldCover']+1)[self.modality_data['ESA_WorldCover'].astype(int).squeeze(1)].transpose(0, 3, 1, 2)
        # self.modality_data['biome'] = np.eye(self.no_data_values['biome']+1)[self.modality_data['biome'].astype(int)]
        # self.modality_data['ecoregion'] = np.eye(self.no_data_values['ecoregion']+1)[self.modality_data['ecoregion'].astype(int)]

        self.tile_count = len(self.modality_data['Sentinel2']) # number of tiles for the task

        print(f'{task} tile count: {self.tile_count}')

        if hasattr(self, 'task_data'):
            print(f'{task}: {self.task_data.shape}')

        for modality, data in self.modality_data.items():
            print(f'{modality}: {data.shape}')

    def __len__(self):
        return self.tile_count

    def __getitem__(self, index):
        # modalities for the tile
        tile_modality_data = {modality: data[index] for modality, data in self.modality_data.items()}

        # normalization
        for modality in ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'precipitation', 'temperature']:
            masked = np.ma.masked_equal(tile_modality_data[modality], self.no_data_values[modality]) # masks the no-data values
            normalized = (masked - self.split_data[f'{modality}_train_means']) / self.split_data[f'{modality}_train_stds'] # normalization
            tile_modality_data[modality] = normalized.filled(0) # replaces NaNs with the post-normalization mean

        # task data for the tile
        if hasattr(self, 'task_data'):
            task_data = self.task_data[index]

            if len(task_data.shape) == 2: # for biomass
                task_data = np.expand_dims(task_data, axis=0)

            return {modality: torch.tensor(data, dtype=torch.float32) for modality, data in tile_modality_data.items()}, torch.tensor(task_data, dtype=torch.float32)
        else:
            return {modality: torch.tensor(data, dtype=torch.float32) for modality, data in tile_modality_data.items()}, {modality: torch.tensor(data, dtype=torch.float32) for modality, data in tile_modality_data.items()}
