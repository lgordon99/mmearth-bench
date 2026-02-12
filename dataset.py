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
import sys
import torch
import utils

data_dir_path = os.environ['DATA_DIR_PATH']

sys.path.append(f'{data_dir_path}/pretrained_checkpoints/galileo/src')
from data.utils import construct_galileo_input, PRETRAINING_NORMALIZING_DICT

# ============================================== GLOBAL VARIABLES ============================================== #

task_modalities = utils.read_json(f'{data_dir_path}/task_modalities.json')
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
normalization_data = utils.read_json(f'{data_dir_path}/normalization_data.json')
axes_to_collapse = (1, 2)

# ============================================== FUNCTIONS ============================================== #

def ensure_2d(array):
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    return array

def make_anysat_s1(sentinel1):
    vv = sentinel1[0]
    vh = sentinel1[1]
    ratio = np.ma.minimum(vv / (vh + 1e-6), 10**6)

    return np.ma.vstack([sentinel1, ratio[np.newaxis, :, :]])

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

def make_galileo_ndvi(nir, red):
    return np.expand_dims(np.where((nir + red) > 0, (nir - red) / (nir + red), 0), 0) # NDVI calculation with handling for division by zero, expanded to have a channel dimension

def get_vv_vh_least_nans(sentinel1):
    asc_num_valid_pixels = sentinel1[:2].count() # counts the number of valid pixels in the ascending VV and VH bands
    desc_num_valid_pixels = sentinel1[2:].count() # counts the number of valid pixels in the descending VV and VH bands
    vv_vh = sentinel1[[0,1]] if asc_num_valid_pixels >= desc_num_valid_pixels else sentinel1[[2, 3]] # extracts the VV and VH bands from either the ascending or descending pass, whichever has more valid pixels

    return vv_vh

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, architecture, adaptation_mode, train_percent):
        self.task = task
        self.architecture = architecture
        self.adaptation_mode = adaptation_mode
        self.train_percent = train_percent
        self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            self.task_modality_data = {modality: h5_file[modality][:] for modality in task_modalities}
            self.ids = h5_file['id'][:]

            if 'ConvNeXtV2A' not in architecture:
                self.input_data = {modality: ensure_2d(h5_file[modality][:]) for modality in normalization_data[self.architecture].keys() if modality in h5_file.keys()}

            if architecture == 'AnySat':
                self.input_data['date'] = ensure_2d(np.array([date(*map(int, sentinel2_date.split('-'))).timetuple().tm_yday - 1 for sentinel2_date in h5_file['sentinel2_date'].asstr()[...]])) # day of year
            elif architecture == 'TerraMind':
                self.input_data['RGB'] = np.ma.stack([make_terramind_rgb(np.ma.masked_equal(sentinel2[[3,2,1]], no_data_values['Sentinel2'])) for sentinel2 in self.input_data['Sentinel2']])
            elif architecture == 'CopernicusFM':
                geolocation = h5_file['geolocation'][:]
                self.input_data['longitude'] = ensure_2d(geolocation[:, 0])
                self.input_data['latitude'] = ensure_2d(geolocation[:, 1])
                self.input_data['time'] = ensure_2d(np.array([(date(*map(int, sentinel2_date.split('-'))) - date(1970, 1, 1)).days for sentinel2_date in h5_file['sentinel2_date'].asstr()[...]])) # number of days after 1/1/1970
            elif architecture == 'Galileo':
                self.input_data['month'] = ensure_2d(np.array([int(sentinel2_date.split('-')[1]) - 1 for sentinel2_date in h5_file['sentinel2_date'].asstr()[...]])) # each month is an integer in the range [0, 11]
                self.input_data['NDVI'] = np.ma.stack([make_galileo_ndvi(*np.ma.masked_equal(sentinel2[[7,3]], no_data_values['Sentinel2'])) for sentinel2 in self.input_data['Sentinel2']])

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

            if 'TTT-Geo' in adaptation_mode:
                self.geolocation = h5_file['geolocation'][:]

        self.tile_count = len(self.task_modality_data['Sentinel2']) # number of tiles for the task

        print(f'{task} tile count: {self.tile_count}')
        print(f'{task}: {self.task_data.shape}')

        if hasattr(self, 'input_data'):
            print('Input data')
            for modality, data in self.input_data.items():
                print(f'{modality}: {data.shape}')

        print('Task modality data')
        for modality, data in self.task_modality_data.items():
            print(f'{modality}: {data.shape}')

    def __len__(self):
        return self.tile_count

    def __getitem__(self, index):
        # modalities for the tile
        tile_task_modality_data = {modality: {'data': data[index]} for modality, data in self.task_modality_data.items()}

        for modality in tile_task_modality_data.keys():
            # compute a valid mask for all modalities that can have NaNs
            if modality not in ['geolocation_encoding', 'month_encoding']:
                tile_task_modality_data[modality]['valid_mask'] = (tile_task_modality_data[modality]['data'] != no_data_values[modality]).squeeze() # mask for where the data is valid

            # normalization
            if modality in ['Sentinel2', 'Sentinel1', 'ASTER_GDEM', 'ETH_GCH', 'precipitation', 'temperature']: # continuous modalities without an encoding
                masked = np.ma.masked_equal(tile_task_modality_data[modality]['data'], no_data_values[modality]) # masks the no-data values
                means = json.loads(json.dumps(self.split_data[f'{modality}_train_{self.train_percent}%_means']).replace('null', '0'))
                stds = json.loads(json.dumps(self.split_data[f'{modality}_train_{self.train_percent}%_stds']).replace('null', '1'))
                normalized = (masked - means) / stds # normalization
                tile_task_modality_data[modality]['data'] = normalized.filled(0) # replaces NaNs with the post-normalization mean

        # convert to tensors
        for modality in tile_task_modality_data.keys():
            tile_task_modality_data[modality]['data'] = torch.tensor(tile_task_modality_data[modality]['data'], dtype=torch.float32) # converts to tensor

        # input data for the tile
        if self.architecture == 'ConvNeXtV2A':
            tile_input_data = {'RGB': tile_task_modality_data['Sentinel2']['data'][[3, 2, 1]]}
        else:
            tile_input_data = {modality: data[index] for modality, data in self.input_data.items()}
            architecture_normalization_data = normalization_data[self.architecture]
            tile_input_data = {modality: data[architecture_normalization_data[modality]['bands']] for modality, data in tile_input_data.items()}

            if 'Galileo' in self.architecture:
                s2_b2, s2_b3, s2_b4, s2_b5, s2_b6, s2_b7, s2_b8, s2_b8A, s2_b11, s2_b12 = np.ma.masked_equal(tile_input_data['Sentinel2'], no_data_values['Sentinel2'])
                s2 = np.expand_dims(np.stack([s2_b2.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][2]),
                                            s2_b3.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][3]),
                                            s2_b4.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][4]),
                                            s2_b5.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][5]),
                                            s2_b6.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][6]),
                                            s2_b7.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][7]),
                                            s2_b8.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][8]),
                                            s2_b8A.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][9]),
                                            s2_b11.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][10]),
                                            s2_b12.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][11])]).transpose(1, 2, 0), axis=2)

                if self.architecture == 'Galileo':
                    s1_vv, s1_vh = get_vv_vh_least_nans(np.ma.masked_equal(tile_input_data['Sentinel1'], no_data_values['Sentinel1']))
                    s1 = np.expand_dims(np.stack([s1_vv.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][0]),
                                                s1_vh.filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][1])]).transpose(1, 2, 0), axis=2)
                    ndvi = np.expand_dims(tile_input_data['NDVI'].filled(PRETRAINING_NORMALIZING_DICT['13']['mean'][12]).transpose(1, 2, 0), axis=2)
                    temperature = np.ma.masked_equal(tile_input_data['temperature'], no_data_values['temperature'])
                    precipitation = np.ma.masked_equal(tile_input_data['precipitation'], no_data_values['precipitation'])
                    era5 = np.stack([temperature.filled(PRETRAINING_NORMALIZING_DICT['6']['mean'][0]),
                                    precipitation.filled(PRETRAINING_NORMALIZING_DICT['6']['mean'][1])]).transpose(1, 0)
                    elevation, slope = np.ma.masked_equal(tile_input_data['ASTER_GDEM'], no_data_values['ASTER_GDEM'])
                    srtm = np.stack([elevation.filled(PRETRAINING_NORMALIZING_DICT['16']['mean'][0]),
                                    slope.filled(PRETRAINING_NORMALIZING_DICT['16']['mean'][1])]).transpose(1, 2, 0)
                    dw = np.eye(no_data_values['DynamicWorld']+1)[tile_input_data['DynamicWorld'].squeeze().astype(int)][:, :, :no_data_values['DynamicWorld']] # one-hot encodes the Dynamic World data, with no-data values removed
                    dw[dw.sum(axis=-1) == 0] = PRETRAINING_NORMALIZING_DICT['16']['mean'][2:11] # fills pixels where Dynamic World data is not available with the means from the Galileo pretraining dataset
                    latlon = tile_input_data['geolocation']
                    dw_static = np.mean(dw, axis=(0, 1))
                    months = tile_input_data['month']
                    masked_output = construct_galileo_input(s1=torch.tensor(s1, dtype=torch.float32),
                                                            s2=torch.tensor(s2, dtype=torch.float32),
                                                            ndvi=torch.tensor(ndvi, dtype=torch.float32),
                                                            era5=torch.tensor(era5, dtype=torch.float32),
                                                            srtm=torch.tensor(srtm, dtype=torch.float32),
                                                            dw=torch.tensor(dw, dtype=torch.float32),
                                                            latlon=torch.tensor(latlon, dtype=torch.float32),
                                                            dw_static=torch.tensor(dw_static, dtype=torch.float32),
                                                            months=torch.tensor(months),
                                                            normalize=True)
                else:
                    masked_output = construct_galileo_input(s2=torch.tensor(s2, dtype=torch.float32),
                                                            normalize=True)

                tile_input_data = {'space_time_input': masked_output.space_time_x.float(),
                                   'space_input': masked_output.space_x.float(),
                                   'time_input': masked_output.time_x.float(),
                                   'static_input': masked_output.static_x.float(),
                                   'space_time_mask': masked_output.space_time_mask,
                                   'space_mask': masked_output.space_mask,
                                   'time_mask': masked_output.time_mask,
                                   'static_mask': masked_output.static_mask,
                                   'month_input': masked_output.months}
            else:
                for modality in tile_input_data.keys():
                    if len(architecture_normalization_data[modality]['means']) > 0: # if the modality has normalization data
                        masked = np.ma.masked_equal(tile_input_data[modality], no_data_values[modality]) if modality in task_modalities else tile_input_data[modality]
                        collapsed_shape = (masked.shape[0],) + (1,) * (masked.ndim - 1) # singleton dimensions for the number of spatial dimensions

                        # min-max normalization
                        if self.architecture == 'ScaleMAE' or 'DINO' in self.architecture:
                            min_value = masked.min(axis=axes_to_collapse).reshape(collapsed_shape)
                            max_value = masked.max(axis=axes_to_collapse).reshape(collapsed_shape)
                            masked = (masked - min_value) / (max_value - min_value)

                        # extract the VV and VH bands from either the ascending or descending pass, whichever has more valid pixels
                        if modality == 'Sentinel1':
                            masked = get_vv_vh_least_nans(masked)

                            if self.architecture == 'AnySat':
                                masked = make_anysat_s1(masked)

                        # mean-std normalization
                        normalized = (masked - np.expand_dims(architecture_normalization_data[modality]['means'], axis=axes_to_collapse)) / np.expand_dims(architecture_normalization_data[modality]['stds'], axis=axes_to_collapse)

                        if self.architecture == 'SatlasNet':
                            normalized = np.ma.clip(normalized, 0, 1) # clips values to [0, 1]

                        tile_input_data[modality] = normalized.filled(0)

                    tile_input_data[modality] = torch.tensor(tile_input_data[modality], dtype=torch.float32) # converts to tensor

        # task data for the tile
        task_data = self.task_data[index]

        if self.task == 'biomass': # for biomass
            task_data = np.expand_dims(task_data, axis=0)

        return tile_input_data, tile_task_modality_data, torch.tensor(task_data, dtype=torch.float32), index, self.ids[index]

if __name__ == '__main__':
    dataset = MMEarthBenchDataset(task='soil_nitrogen', architecture='Galileo', adaptation_mode='FT', train_percent='5')
