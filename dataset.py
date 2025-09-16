'''
dataset.py
'''

# ============================================== IMPORTS ============================================== #

from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import torch
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = os.environ['DATA_DIR_PATH']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, adaptation_mode):
        self.task = task
        self.adaptation_mode = adaptation_mode
        self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            self.modality_data = {modality: h5_file[modality][:] for modality in no_data_values.keys()}
            self.task_data = h5_file[task][:]

        # if adaptation_mode == 'multimodal':
        #     # convert categorical modalities to one-hot encoding
        #     self.modality_data['DynamicWorld'] = np.eye(self.no_data_values['DynamicWorld']+1)[self.modality_data['DynamicWorld'].astype(int).squeeze(1)].transpose(0, 3, 1, 2)
        #     self.modality_data['ESA_WorldCover'] = np.eye(self.no_data_values['ESA_WorldCover']+1)[self.modality_data['ESA_WorldCover'].astype(int).squeeze(1)].transpose(0, 3, 1, 2)
        #     self.modality_data['biome'] = np.eye(self.no_data_values['biome']+1)[self.modality_data['biome'].astype(int)]
        #     self.modality_data['ecoregion'] = np.eye(self.no_data_values['ecoregion']+1)[self.modality_data['ecoregion'].astype(int)]

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
        tile_modality_data = {modality: {'data': data[index]} for modality, data in self.modality_data.items()}

        for modality in tile_modality_data.keys():
            # compute a valid mask for all modalities that can have NaNs
            if modality not in ['geolocation', 'month']:
                tile_modality_data[modality]['valid_mask'] = (tile_modality_data[modality]['data'] != no_data_values[modality]).squeeze() # mask for where the data is valid

            # normalization
            if modality in ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'precipitation', 'temperature']: # continuous modalities without an encoding
                masked = np.ma.masked_equal(tile_modality_data[modality]['data'], no_data_values[modality]) # masks the no-data values
                normalized = (masked - self.split_data[f'{modality}_train_means']) / self.split_data[f'{modality}_train_stds'] # normalization
                tile_modality_data[modality]['data'] = normalized.filled(0) # replaces NaNs with the post-normalization mean

        if self.adaptation_mode == 'multimodal':
            # convert categorical modalities to one-hot encoding
            tile_modality_data['DynamicWorld']['data'] = np.eye(no_data_values['DynamicWorld']+1)[tile_modality_data['DynamicWorld']['data'].astype(int)].squeeze().transpose(2, 0, 1)
            tile_modality_data['ESA_WorldCover']['data'] = np.eye(no_data_values['ESA_WorldCover']+1)[tile_modality_data['ESA_WorldCover']['data'].astype(int)].squeeze().transpose(2, 0, 1)
            tile_modality_data['biome']['data'] = np.eye(no_data_values['biome']+1)[tile_modality_data['biome']['data'].astype(int)]
            tile_modality_data['ecoregion']['data'] = np.eye(no_data_values['ecoregion']+1)[tile_modality_data['ecoregion']['data'].astype(int)]

        # convert to tensors
        for modality in tile_modality_data.keys():
            tile_modality_data[modality]['data'] = torch.tensor(tile_modality_data[modality]['data'], dtype=torch.float32) # converts to tensor

        # task data for the tile
        task_data = self.task_data[index]

        if len(task_data.shape) == 2: # for biomass
            task_data = np.expand_dims(task_data, axis=0)

        return tile_modality_data, torch.tensor(task_data, dtype=torch.float32)
