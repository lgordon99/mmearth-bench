# requires 50GB memory to run

# imports
from collections import defaultdict
from sys import argv
import h5py
import numpy as np
import os
import rasterio
import subprocess
import utils

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETHGCH', 'DynamicWorld', 'ESA_Worldcover', 'SCL', 'MSK_CLDPRB', 'QA60', 'S2CLOUDLESS']
image_level_modalities = ['climate', 'latitude', 'longitude', 'month', 'biome', 'ecoregion', 'MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', 'S2CLOUDLESS_CLOUDY_PIXEL_FRACTION']
no_data_values = {'Sentinel1': float('-inf'),
                  'climate': float('inf'),
                  'latitude': float('-inf'),
                  'longitude': float('-inf'),
                  'month': float('-inf'),
                  'biome': 255,
                  'ecoregion': 65535}

def get_tag_value(tags, key):
    value = tags[key]

    return no_data_values[[modality for modality in no_data_values.keys() if modality in key][0]] if value == 'None' else value

def check_is_number(value):
    try:
        float(value)
        return True
    except:
        return False

def convert_tiffs_to_h5(task):
    task_data_dir = f'{data_dir_path}/{task}/data'
    data = defaultdict(list) # dictionary whose default value is an empty list

    with h5py.File(f'{data_dir_path}/{task}/{task}_h5.hdf5', 'w') as h5_file:
        for tiff in os.listdir(task_data_dir):
            with rasterio.open(f'{task_data_dir}/{tiff}') as tiff:
                array = tiff.read()
                band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)}
                tags = tiff.tags()

                # pixel-level modalities
                for modality in pixel_level_modalities:
                    modality_band_numbers = [band_number for band_number, band_name in band_names.items() if modality in band_name]
                    modality_array = array[modality_band_numbers]

                    if modality == 'Sentinel1':
                        for band_number in range(len(modality_array)):
                            if np.all(modality_array[band_number] == -9999):
                                modality_array[band_number] = np.full(modality_array[band_number].shape, no_data_values['Sentinel1'])

                    data[modality].append(modality_array)

                # image-level modalities
                for modality in image_level_modalities:
                    data[modality].append(np.array([value for key, value in ((key, get_tag_value(tags, key)) for key in tags.keys()) if modality in key.split('_')[0] and check_is_number(value)]).astype('float32'))

                # task data
                if task == 'biomass':
                    data[task].append(array[29])
                elif task == 'species':
                    species = [int(value) for value in get_tag_value(tags, task).split(',')]
                    species_vector = np.zeros(100)
                    species_vector[species] = 1
                    data[task].append(species_vector)
                elif 'soil' in task:
                    data[task].append(np.array([tags[task]]).astype('float32'))

                # geographic data
                data['crs'] = tiff.crs.to_string()
                data['transform'] = np.array([i for i in tiff.transform])
                assert all(data['transform'][i] == tiff.transform[i] for i in range(len(data['transform'])))

        for key, value in data.items():
            if key != 'crs':
                print(key, np.array(value).shape)

                h5_file.create_dataset(key, data=np.array(value))
            else:
                print(key, len(value))

                h5_file.create_dataset(key, data=value)

if __name__ == '__main__':
    if 'for' not in argv[1]: # python convert_to_h5.py TASK
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        subprocess.run(['sbatch', '-t', '0-01:00:00', '-p', partitions, '--mem', '50G', '--job-name', f'{argv[1]}_convert_to_h5', '-o', f'bash-outputs/{argv[1]}_convert_to_h5.out', '-e', f'bash-errors/{argv[1]}_convert_to_h5.err', 'job.sh', env_path, 'convert_to_h5.py', f'for_{argv[1]}'])
    else: # python generate_map_data.py for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        convert_tiffs_to_h5(task)
