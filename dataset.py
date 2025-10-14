'''
dataset.py
'''

# ============================================== IMPORTS ============================================== #

from datetime import date
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
normalization_data = utils.read_json(f'{data_dir_path}/normalization_data.json')
axes_to_collapse = (1, 2)

# ============================================== FUNCTIONS ============================================== #

def ensure_2d(array):
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    return array

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

def get_vv_vh_least_nans(sentinel1):
    asc_num_valid_pixels = sentinel1[:2].count() # counts the number of valid pixels in the ascending VV and VH bands
    desc_num_valid_pixels = sentinel1[2:].count() # counts the number of valid pixels in the descending VV and VH bands
    vv_vh = sentinel1[[0,1]] if asc_num_valid_pixels >= desc_num_valid_pixels else sentinel1[[2, 3]] # extracts the VV and VH bands from either the ascending or descending pass, whichever has more valid pixels

    return vv_vh

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, architecture, adaptation_mode):
        self.task = task
        self.architecture = architecture
        self.adaptation_mode = adaptation_mode
        self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            self.input_data = {modality: ensure_2d(h5_file[modality][:]) for modality in normalization_data[self.architecture].keys() if modality in h5_file.keys()}
            self.task_modality_data = {modality: h5_file[modality][:] for modality in no_data_values.keys()}

            if architecture == 'TerraMind':
                self.input_data['RGB'] = np.ma.stack([make_terramind_rgb(np.ma.masked_equal(sentinel2[[3,2,1]], no_data_values['Sentinel2'])) for sentinel2 in self.input_data['Sentinel2']])
                # train_images = self.task_modality_data['rgb'][self.split_data['train_indices']]
                # axes_to_collapse = tuple(i for i in range(train_images.ndim) if i != 1)
                # collapsed_shape = (train_images.shape[1],) + (1,) * (train_images.ndim - 2)
                # self.rgb_train_means = train_images.mean(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
                # self.rgb_train_stds = train_images.std(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
            elif architecture == 'CopernicusFM':
                # self.task_modality_data['longitude'] = h5_file['longitude'][:]
                # self.task_modality_data['latitude'] = h5_file['latitude'][:]
                self.input_data['Time'] = ensure_2d(np.array([(date(*map(int, sentinel2_date.split('-'))) - date(1970, 1, 1)).days for sentinel2_date in h5_file['sentinel2_date'].asstr()[...]])) # number of days after 1/1/1970

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

        self.tile_count = len(self.task_modality_data['Sentinel2']) # number of tiles for the task

        print(f'{task} tile count: {self.tile_count}')
        print(f'{task}: {self.task_data.shape}')

        print('Input data')
        for modality, data in self.input_data.items():
            print(f'{modality}: {data.shape}')

        print('Task modality data')
        for modality, data in self.task_modality_data.items():
            print(f'{modality}: {data.shape}')

    def __len__(self):
        return self.tile_count

    def __getitem__(self, index):
        # input data for the tile
        tile_input_data = {modality: data[index] for modality, data in self.input_data.items()}
        architecture_normalization_data = normalization_data[self.architecture]
        tile_input_data = {modality: data[architecture_normalization_data[modality]['bands']] for modality, data in tile_input_data.items()}

        for modality in tile_input_data.keys():
            if len(architecture_normalization_data[modality]['means']) > 0: # if the modality has normalization data
                masked = np.ma.masked_equal(tile_input_data[modality], no_data_values[modality]) if modality in no_data_values.keys() else tile_input_data[modality]
                collapsed_shape = (masked.shape[0],) + (1,) * (masked.ndim - 1) # singleton dimensions for the number of spatial dimensions

                # min-max normalization
                if self.architecture == 'ScaleMAE' or 'DINO' in self.architecture:
                    min_value = masked.min(axis=axes_to_collapse).reshape(collapsed_shape)
                    max_value = masked.max(axis=axes_to_collapse).reshape(collapsed_shape)
                    masked = (masked - min_value) / (max_value - min_value)

                # extract the VV and VH bands from either the ascending or descending pass, whichever has more valid pixels
                if modality == 'Sentinel1':
                    masked = get_vv_vh_least_nans(masked)
                    collapsed_shape = (masked.shape[0],) + (1,) * (masked.ndim - 1) # singleton dimensions for the number of spatial dimensions

                # mean-std normalization
                normalized = (masked - np.expand_dims(architecture_normalization_data[modality]['means'], axis=axes_to_collapse)) / np.expand_dims(architecture_normalization_data[modality]['stds'], axis=axes_to_collapse)
                post_normalization_means = normalized.mean(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
                tile_input_data[modality] = normalized.filled(post_normalization_means)

            tile_input_data[modality] = torch.tensor(tile_input_data[modality], dtype=torch.float32) # converts to tensor

        # modalities for the tile
        tile_task_modality_data = {modality: {'data': data[index]} for modality, data in self.task_modality_data.items()}

        for modality in tile_task_modality_data.keys():
            # compute a valid mask for all modalities that can have NaNs
            if modality not in ['geolocation', 'month']:
                tile_task_modality_data[modality]['valid_mask'] = (tile_task_modality_data[modality]['data'] != no_data_values[modality]).squeeze() # mask for where the data is valid

            # normalization
            if modality in ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'precipitation', 'temperature']: # continuous modalities without an encoding
                masked = np.ma.masked_equal(tile_task_modality_data[modality]['data'], no_data_values[modality]) # masks the no-data values
                normalized = (masked - self.split_data[f'{modality}_train_means']) / self.split_data[f'{modality}_train_stds'] # normalization
                tile_task_modality_data[modality]['data'] = normalized.filled(0) # replaces NaNs with the post-normalization mean
            # elif modality == 'rgb':
            #     normalized = (tile_task_modality_data[modality]['data'] - self.rgb_train_means) / self.rgb_train_stds # normalization
            #     tile_task_modality_data[modality]['data'] = normalized.filled(0) # replaces NaNs with the post-normalization mean

        if self.adaptation_mode in ['multimodal', 'multimodal_joint_training', 'ttt-mjt', 'multimodal_mt3', 'multimodal_sln', 'maml_encode']:
            # convert categorical modalities to one-hot encoding
            dynamic_world_onehot = np.eye(no_data_values['DynamicWorld']+1)[tile_task_modality_data['DynamicWorld']['data'].astype(int)].squeeze().transpose(2, 0, 1)
            esa_worldcover_onehot = np.eye(no_data_values['ESA_WorldCover']+1)[tile_task_modality_data['ESA_WorldCover']['data'].astype(int)].squeeze().transpose(2, 0, 1)
            biome_onehot = np.eye(no_data_values['biome']+1)[tile_task_modality_data['biome']['data'].astype(int)]
            ecoregion_onehot = np.eye(no_data_values['ecoregion']+1)[tile_task_modality_data['ecoregion']['data'].astype(int)]

            if self.adaptation_mode == 'multimodal':
                tile_task_modality_data['DynamicWorld']['data'] = dynamic_world_onehot
                tile_task_modality_data['ESA_WorldCover']['data'] = esa_worldcover_onehot
                tile_task_modality_data['biome']['data'] = biome_onehot
                tile_task_modality_data['ecoregion']['data'] = ecoregion_onehot
            elif self.adaptation_mode in ['multimodal_joint_training', 'ttt-mjt', 'multimodal_mt3', 'multimodal_sln']:
                tile_task_modality_data['DynamicWorld_onehot'] = {'data': dynamic_world_onehot}
                tile_task_modality_data['ESA_WorldCover_onehot'] = {'data': esa_worldcover_onehot}
                tile_task_modality_data['biome_onehot'] = {'data': biome_onehot}
                tile_task_modality_data['ecoregion_onehot'] = {'data': ecoregion_onehot}

        # convert to tensors
        for modality in tile_task_modality_data.keys():
            tile_task_modality_data[modality]['data'] = torch.tensor(tile_task_modality_data[modality]['data'], dtype=torch.float32) # converts to tensor

        # task data for the tile
        task_data = self.task_data[index]

        if self.task == 'biomass': # for biomass
            task_data = np.expand_dims(task_data, axis=0)

        return tile_input_data, tile_task_modality_data, torch.tensor(task_data, dtype=torch.float32)
