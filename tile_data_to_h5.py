# imports
from utils import read_json, read_yaml
import numpy as np
import os
import rasterio

IMAGE_SIZE = read_yaml('config.yml')['IMAGE_SIZE']
task = 'biomass'
tile_image_level_data = read_json(f'tiles/{task}/tile_image_level_data.json')
bands_by_modality = {'pixel_level': {'sentinel2': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B11', 'B12', 'SCL', 'MSK_CLDPRB', 'QA60'],
                                     'sentinel1': ['asc_VV', 'asc_VH', 'asc_HH', 'asc_HV', 'desc_VV', 'desc_VH', 'desc_HH', 'desc_HV'],
                                     'aster': ['elevation', 'slope'],
                                     'eth': ['height', 'uncertainty'],
                                     'dynamic_world': ['class'],
                                     'esa_worldcover': ['class']},
                     task: ['label']}
bands = np.array([value for values in bands_by_modality['pixel_level'].values() for value in values])
tile_ids = [file.split('_')[1] for file in os.listdir(f'tiles/{task}/pixel_level_data') if file.endswith('.tif')]

for tile_id in tile_ids:
    # print(tile_id)
    pixel_level_data = {}

    with rasterio.open(f'tiles/{task}/pixel_level_data/tile_{tile_id}_pixel_level_data.tif') as tiff:
        array = tiff.read().transpose(1, 2, 0)
        start_col, start_row = (np.array(array.shape)[:2] - IMAGE_SIZE) // 2 # center crop
        array = array[start_row : start_row + IMAGE_SIZE, start_col : start_col + IMAGE_SIZE].transpose(2, 0, 1)

        for modality in bands_by_modality['pixel_level'].keys():
            pixel_level_data[modality] = array[np.where(bands == bands_by_modality['pixel_level'][modality][0])[0][0] : np.where(bands == bands_by_modality['pixel_level'][modality][-1])[0][0]+1]

    dw = pixel_level_data['dynamic_world']
    print(len(dw[dw == 0]))
    # break
    image_level_data = {}
    # image_level_data['biome'] = tile_image_level_data[tile_id][]

    # era_data = list(tile_image_level_data[tile_id]['era5'].values())
    # image_level_data['era5'] = np.stack([era_data['month1'] + era_data['month2'] + era_data['year']], axis=0).astype('float32'))
    # break