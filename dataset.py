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
            self.sentinel1 = h5_file['Sentinel1'][:]
            self.asterdem = h5_file['AsterDEM'][:]
            self.ethgch = h5_file['ETHGCH'][:]
            self.dynamicworld = h5_file['DynamicWorld'][:]
            self.esaworldcover = h5_file['ESA_Worldcover'][:]
            self.precipitation = h5_file['climate'][:, :3] # first 9 bands are precipitation
            self.temperature = h5_file['climate'][:, 3:] # last 3 bands are temperature
            self.geolocation = np.concatenate([h5_file['longitude'][:], h5_file['latitude'][:]], axis=1)
            self.month = h5_file['month'][:]
            self.biome = h5_file['biome'][:]
            self.ecoregion = h5_file['ecoregion'][:]
            self.task_data = h5_file[task][:]
            self.tile_count = len(self.sentinel2) # number of tiles for the task

            print(f'{task} tile count: {self.tile_count}')
            print(f'Sentinel-2: {self.sentinel2.shape}')
            print(f'Sentinel-1: {self.sentinel1.shape}')
            print(f'AsterDEM: {self.asterdem.shape}')
            print(f'ETHGCH: {self.ethgch.shape}')
            print(f'DynamicWorld: {self.dynamicworld.shape}')
            print(f'ESA_Worldcover: {self.esaworldcover.shape}')
            print(f'Precipitation: {self.precipitation.shape}')
            print(f'Temperature: {self.temperature.shape}')
            print(f'Geolocation: {self.geolocation.shape}')
            print(f'Month: {self.month.shape}')
            print(f'Biome: {self.biome.shape}')
            print(f'Ecoregion: {self.ecoregion.shape}')
            print(f'{task}: {self.task_data.shape}')

        self.split_data = utils.read_json(f'{data_dir_path}/{task}/{task}_split_data.json')

        # FIX
        # if task == 'species':
        #     train_species_counts = task_data[train_indices].sum(axis=0)
        #     species_with_too_few_train_observations = np.where(train_species_counts < 50)[0]
        #     task_data[np.ix_(test_indices, species_with_too_few_train_observations)] = 0

    def __len__(self):
        return self.tile_count

    def __getitem__(self, index):
        # modalities for the tile
        modality_data = {'sentinel2': self.sentinel2[index],
                         'sentinel1': self.sentinel1[index],
                         'asterdem': self.asterdem[index],
                         'ethgch': self.ethgch[index],
                         'dynamicworld': self.dynamicworld[index],
                         'esaworldcover': self.esaworldcover[index],
                         'precipitation': self.precipitation[index],
                         'temperature': self.temperature[index],
                         'geolocation': self.geolocation[index],
                         'month': self.month[index],
                         'biome': self.biome[index],
                         'ecoregion': self.ecoregion[index]}

        # normalization
        modality_data['sentinel2'] = (modality_data['sentinel2'] - self.split_data['train_band_means']) / self.split_data['train_band_stds'] # normalization

        # task data for the tile
        task_data = self.task_data[index]

        if len(task_data.shape) == 2: # for biomass
            task_data = np.expand_dims(task_data, axis=0)

        return {modality: torch.tensor(data, dtype=torch.float32) for modality, data in modality_data.items()}, torch.tensor(task_data, dtype=torch.float32)
