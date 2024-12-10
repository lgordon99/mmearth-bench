# imports
from collections import defaultdict
import h5py
import numpy as np
import os
import rasterio
import utils

IMAGE_SIZE = utils.read_yaml('config.yml')['IMAGE_SIZE']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETHGCH', 'DynamicWorld', 'ESA_Worldcover', 'SCL', 'MSK_CLDPRB', 'QA60']
# climate_bands = ['climate_temperature_last_month_mean',
#                  'climate_temperature_last_month_min',
#                  'climate_temperature_last_month_max',
#                  'climate_precipitation_last_month',
#                  'climate_temperature_month_mean',
#                  ]
no_data_values = {}

def get_tag_value(tags, key):
    value = tags[key]

    return no_data_values[key.split('_')[0]] if value is None else value

def convert_tiffs_to_h5(task):
    data = defaultdict(list) # dictionary whose default value is an empty list

    for tiff in os.listdir(f'{task}/data'):
        with rasterio.open(f'{task}/data/{tiff}') as tiff:
            array = tiff.read().transpose(1, 2, 0) # shape (number of bands, number of rows, number of columns)
            start_col, start_row = (np.array(array.shape)[:2] - IMAGE_SIZE) // 2
            array = array[start_row : start_row + IMAGE_SIZE, start_col : start_col + IMAGE_SIZE].transpose(2, 0, 1) # center crop of shape (number of bands, number of rows, number of columns)
            band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)}
            tags = tiff.tags()

            with h5py.File(f'{task}/{task}_h5.hdf5', 'w') as h5_file:
                for modality in pixel_level_modalities:
                    band_numbers = [band_number for band_number, band_name in band_names.items() if modality in band_name]
                    modality_array = array[band_numbers]
                    # h5_file.create_dataset(modality, data=modality_array)
                    data[modality].append(modality_array)

                # h5_file.create_dataset('climate', np.array([tags['climate_temperature_last_month_mean'],
                #                                             tags['climate_temperature_last_month_min'],
                #                                             tags['climate_temperature_last_month_max'],
                #                                             tags['climate_precipitation_last_month'],
                #                                             tags['climate_temperature_month_mean'],
                #                                             tags['climate_temperature_month_min'],
                #                                             tags['climate_temperature_month_max'],
                #                                             tags['climate_precipitation_month'],
                #                                             tags['climate_temperature_year_mean'],
                #                                             tags['climate_temperature_year_min'],
                #                                             tags['climate_temperature_year_max'],
                #                                             tags['climate_precipitation_year']]).astype('float32'))
                data['climate'].append(np.array([get_tag_value(tags, key) for key in tags.keys() if 'climate' in key]).astype('float32'))
                # data['climate'].append(np.array([get_tag_value(tags, 'climate_temperature_last_month_mean'),
                #                                 get_tag_value(tags, 'climate_temperature_last_month_min'),
                #                                 get_tag_value(tags, 'climate_temperature_last_month_max'),
                #                                 get_tag_value(tags, 'climate_precipitation_last_month'),
                #                                 get_tag_value(tags, 'climate_temperature_month_mean'),
                #                                 get_tag_value(tags, 'climate_temperature_month_min'),
                #                                 get_tag_value(tags, 'climate_temperature_month_max'),
                #                                 get_tag_value(tags, 'climate_precipitation_month'),
                #                                 tags['climate_temperature_year_mean'],
                #                                 tags['climate_temperature_year_min'],
                #                                 tags['climate_temperature_year_max'],
                #                                 tags['climate_precipitation_year']]).astype('float32'))
                # h5_file.create_dataset('latitude', np.array([tags['latitude_sin'], tags['latitude_cos']]).astype('float32'))
                data['latitude'].append(np.array([tags['latitude_sin'], tags['latitude_cos']]).astype('float32'))
                
                # h5_file.create_dataset('longitude', np.array([tags['longitude_sin'], tags['longitude_cos']]).astype('float32'))
                data['longitude'].append(np.array([tags['longitude_sin'], tags['longitude_cos']]).astype('float32'))
                
                # h5_file.create_dataset('month', np.array([tags['month_sin'], tags['month_cos']]).astype('float32'))
                data['month'].append(np.array([tags['month_sin'], tags['month_cos']]).astype('float32'))

                biome_labels = utils.read_json('biomes_ecoregions_data/biome_labels.json')
                # h5_file.create_dataset('biome', np.array([biome_labels[tags['biome']]]))
                data['biome'].append(np.array([biome_labels[tags['biome']]]))

                ecoregion_labels = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')
                # h5_file.create_dataset('ecoregion', np.array([ecoregion_labels[tags['ecoregion']]]))
                data['ecoregion'].append(np.array([ecoregion_labels[tags['ecoregion']]]))

                # exit()
                # for band in range(1, array.shape[0] + 1):
                #     band_name = tiff.tags(band)['BAND_NAME']

            #         if 
            #     exit()
            #     h5_file.create_dataset()

        #     for modality in bands_by_modality['pixel_level'].keys():
        #         pixel_level_data[modality] = array[np.where(bands == bands_by_modality['pixel_level'][modality][0])[0][0] : np.where(bands == bands_by_modality['pixel_level'][modality][-1])[0][0]+1]

        # dw = pixel_level_data['dynamic_world']
        # print(len(dw[dw == 0]))
        # # break
        # image_level_data = {}
        # image_level_data['biome'] = tile_image_level_data[tile_id][]

        # era_data = list(tile_image_level_data[tile_id]['era5'].values())
        # image_level_data['era5'] = np.stack([era_data['month1'] + era_data['month2'] + era_data['year']], axis=0).astype('float32'))
        # break

def open_h5(task):
    with h5py.File(f'{task}/{task}_h5.hdf5', 'r') as h5_file:
        # print(h5_file.keys())
        # print(h5_file['Sentinel1'].shape)
        print(h5_file['biome'][()])

convert_tiffs_to_h5('soil_nitrogen')

# open_h5('soil_nitrogen')