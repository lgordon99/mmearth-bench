# imports
from collections import defaultdict
from tkinter import E
import h5py
import numpy as np
import os
import rasterio
import utils

IMAGE_SIZE = utils.read_yaml('config.yml')['IMAGE_SIZE']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETHGCH', 'DynamicWorld', 'ESA_Worldcover', 'SCL', 'MSK_CLDPRB', 'QA60']
image_level_modalities = ['climate', 'latitude', 'longitude', 'month', 'biome', 'ecoregion']
# climate_bands = ['climate_temperature_last_month_mean',
#                  'climate_temperature_last_month_min',
#                  'climate_temperature_last_month_max',
#                  'climate_precipitation_last_month',
#                  'climate_temperature_month_mean',
#                  ]
no_data_values = {'Sentinel1': float('-inf'),
                  'climate': float('inf'),
                  'latitude': float('-inf'),
                  'longitude': float('-inf'),
                  'month': float('-inf'),
                  'biome': 255,
                  'ecoregion': 65535}

def get_tag_value(tags, key):
    value = tags[key]

    return no_data_values[key.split('_')[0]] if value == 'None' else value.split(',')[0]

def check_is_number(value):
    try:
        float(value)
        return True
    except:
        return False

def convert_tiffs_to_h5(task):
    data = defaultdict(list) # dictionary whose default value is an empty list

    with h5py.File(f'{task}/{task}_h5.hdf5', 'w') as h5_file:
        for tiff in os.listdir(f'{task}/data'):
            with rasterio.open(f'{task}/data/{tiff}') as tiff:
                array = tiff.read().transpose(1, 2, 0) # shape (number of bands, number of rows, number of columns)
                start_col, start_row = (np.array(array.shape)[:2] - IMAGE_SIZE) // 2
                array = array[start_row : start_row + IMAGE_SIZE, start_col : start_col + IMAGE_SIZE].transpose(2, 0, 1) # center crop of shape (number of bands, number of rows, number of columns)
                band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)}
                tags = tiff.tags()

                for modality in pixel_level_modalities:
                    band_numbers = [band_number for band_number, band_name in band_names.items() if modality in band_name]
                    modality_array = array[band_numbers]

                    if modality == 'Sentinel1':
                        for band_number in range(len(modality_array)):
                            if np.all(modality_array[band_number] == -9999):
                                modality_array[band_number] = np.full(modality_array[band_number].shape, no_data_values['Sentinel1'])

                    data[modality].append(modality_array)

                for modality in image_level_modalities:
                    data[modality].append(np.array([value for key, value in ((key, get_tag_value(tags, key)) for key in tags.keys()) if modality in key.split('_')[0] and check_is_number(value)]).astype('float32'))

        for modality, array in data.items():
            h5_file.create_dataset(modality, data=np.array(array))

            print(modality, np.array(array).shape)

def open_h5(task):
    with h5py.File(f'{task}/{task}_h5.hdf5', 'r') as h5_file:
        # print(h5_file.keys())
        # print(h5_file['Sentinel1'].shape)
        print(len(h5_file['biome'][()]))

if __name__ == '__main__':
    # convert_tiffs_to_h5('soil_nitrogen')

    open_h5('soil_nitrogen')