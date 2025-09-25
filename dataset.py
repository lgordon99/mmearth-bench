'''
dataset.py
'''

# ============================================== IMPORTS ============================================== #

from torch.utils.data import Dataset
import h5py
import json
import numpy as np
import os
import torch
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = os.environ['DATA_DIR_PATH']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')

# ============================================== FUNCTIONS ============================================== #

def make_terramind_rgb(rgb):
    rgb = rgb - 1000
    Q2, Q98 = np.quantile(rgb.compressed(), [0.02, 0.98]) # 2nd and 98th percentiles
    rgb = np.ma.where(rgb >= Q2, rgb, Q2 + (rgb - Q2) * 0.5) # compresses values below the 2nd percentile
    rgb = np.ma.where(rgb <= Q98, rgb, Q98 + (rgb - Q98) * 0.5) # compresses values above the 98th percentile
    Q02, Q50, Q998 = np.quantile(rgb.compressed(), [0.002, 0.5, 0.998]) # 0.2nd, 50th, and 99.8th percentiles
    U = max(2000, Q998) # upper bound for normalization
    L = 0 if Q50 < 1000 else Q02 # lower bound for normalization
    rgb = (rgb - L) / (U - L) * 255 # normalization to [0, 255]
    rgb = np.ma.clip(rgb, 0, 255) # clips values to [0, 255]

    return rgb

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, architecture, adaptation_mode):
        self.task = task
        self.architecture = architecture
        self.adaptation_mode = adaptation_mode
        self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            self.modality_data = {modality: h5_file[modality][:] for modality in no_data_values.keys()}

            if architecture == 'TerraMind':
                # rgb = self.modality_data['Sentinel2'][:, [3,2,1]]
                # print(np.count_nonzero(rgb[202] == no_data_values['Sentinel2']))
                # arr1 = np.where(rgb[202] == no_data_values['Sentinel2'])
                # output = make_terramind_rgb(np.ma.masked_equal(rgb[202], no_data_values['Sentinel2']))
                # print(np.count_nonzero(output.filled(no_data_values['Sentinel2']) == no_data_values['Sentinel2']))
                # arr2 = np.where(output.filled(no_data_values['Sentinel2']) == no_data_values['Sentinel2'])
                # print(np.array_equal(arr1, arr2))
                # print(arr1)
                # print(arr2)
                # exit()
                self.modality_data['rgb'] = np.ma.stack([make_terramind_rgb(np.ma.masked_equal(sentinel2[[3,2,1]], no_data_values['Sentinel2'])) for sentinel2 in self.modality_data['Sentinel2']])
                train_images = self.modality_data['rgb'][self.split_data['train_indices']]
                axes_to_collapse = tuple(i for i in range(train_images.ndim) if i != 1)
                collapsed_shape = (train_images.shape[1],) + (1,) * (train_images.ndim - 2)
                self.rgb_train_means = train_images.mean(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
                self.rgb_train_stds = train_images.std(axis=axes_to_collapse).reshape(collapsed_shape).tolist()

            if task == 'species':
                species_list_strings = [json.loads(lst) for lst in h5_file[task].asstr()[...]] # list of lists containing the names of the species in each tile
                species_labels = utils.read_json(f'{data_dir_path}/species/species_labels.json') # dictionary mapping species names to integer labels
                species_list_ints = [[species_labels[species] for species in lst] for lst in species_list_strings] # list of lists containing the integer labels of the species in each tile
                self.task_data = np.zeros((len(species_list_ints), len(species_labels))) # empty multi-label binary matrix for species presence

                for tile_idx in range(len(self.task_data)): # for each tile
                    for species_idx in species_list_ints[tile_idx]: # for each species in the tile
                        self.task_data[tile_idx][species_idx] = 1 # marks the species as present in the tile
            else:
                self.task_data = h5_file[task][:]

        self.tile_count = len(self.modality_data['Sentinel2']) # number of tiles for the task

        print(f'{task} tile count: {self.tile_count}')
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
            if modality not in ['geolocation', 'month', 'rgb']:
                tile_modality_data[modality]['valid_mask'] = (tile_modality_data[modality]['data'] != no_data_values[modality]).squeeze() # mask for where the data is valid

            # normalization
            if modality in ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'precipitation', 'temperature']: # continuous modalities without an encoding
                masked = np.ma.masked_equal(tile_modality_data[modality]['data'], no_data_values[modality]) # masks the no-data values
                normalized = (masked - self.split_data[f'{modality}_train_means']) / self.split_data[f'{modality}_train_stds'] # normalization
                tile_modality_data[modality]['data'] = normalized.filled(0) # replaces NaNs with the post-normalization mean
            elif modality == 'rgb':
                normalized = (tile_modality_data[modality]['data'] - self.rgb_train_means) / self.rgb_train_stds # normalization
                tile_modality_data[modality]['data'] = normalized.filled(0) # replaces NaNs with the post-normalization mean

        if self.adaptation_mode in ['multimodal', 'maml_encode']:
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

        if self.task == 'biomass': # for biomass
            task_data = np.expand_dims(task_data, axis=0)

        return tile_modality_data, torch.tensor(task_data, dtype=torch.float32)
