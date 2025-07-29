# ============================================== IMPORTS ============================================== #

from collections import defaultdict
from sys import argv
import h5py
import json
import numpy as np
import os
import rasterio
import subprocess
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'DynamicWorld', 'ESA_WorldCover', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL']
tile_level_modalities = ['precipitation', 'temperature', 'geolocation', 'month', 'biome', 'ecoregion', 'latitude', 'longitude', 'MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', 'S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', 'SCL_NO_DATA_PIXEL_FRACTION']

# ============================================== FUNCTIONS ============================================== #

def get_tile_id(filename):
    return int(filename.split('_')[1].split('.')[0]) # extracts the tile ID from the TIFF name

def convert_tiffs_to_h5(task):
    start_time = time.time()
    task_tiff_dir = f'{data_dir_path}/{task}/tiffs' # folder where the TIFFs are stored
    data = defaultdict(list) # dictionary whose default value is an empty list

    with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'w') as h5_file:
        for tiff_filename in sorted(os.listdir(task_tiff_dir), key=get_tile_id): # sorts the TIFFs by their IDs
            with rasterio.open(f'{task_tiff_dir}/{tiff_filename}') as tiff: # opens the TIFF
                array = tiff.read() # reads the TIFF as a numpy array
                band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)} # dictionary mapping an index to every band name
                tags = tiff.tags()

                # pixel-level modalities
                for modality in pixel_level_modalities:
                    modality_band_numbers = [band_number for band_number, band_name in band_names.items() if modality in band_name] # extracts the band numbers for the modality
                    data[modality].append(array[modality_band_numbers]) # saves the modality data

                # tile-level modalities
                for modality in tile_level_modalities:
                    data[modality].append(json.loads(tags[modality]))

                # task data
                if task == 'biomass':
                    biomass = array[[band_number for band_number, band_name in band_names.items() if 'biomass' in band_name][0]] # extracts the biomass data
                    data[task].append(biomass) # saves the biomass array
                elif 'soil' in task:
                    data[task].append([float(tags[task])])

                # additional tile data
                data['id'].append(get_tile_id(tiff_filename))
                data['sentinel2_date'].append(tags['sentinel2_date'])
                data['crs'].append(tiff.crs.to_string())
                data['transform'].append([i for i in tiff.transform])
                data['missing_modalities'].append(tags['missing_modalities'])

        for key, value in data.items():
            # print(key, np.array(value).shape)

            # h5_file.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=9)
            h5_file.create_dataset(key, data=value, compression='gzip', compression_opts=9)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

if __name__ == '__main__':
    if 'for' not in argv[1]: # python convert_to_h5.py TASK
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        mem = 300 if argv[1] == 'biomass' else 60
        task = argv[1]
        subprocess.run(['sbatch', '-t', '15:00:00', '-p', partitions, '--mem', f'{mem}G', '--job-name', f'{task}_convert_to_h5', '-o', f'{data_dir_path}/{task}/output-files/{task}_convert_to_h5.out', '--account', 'gajos_lab', 'job.sh', env_path, 'convert_to_h5.py', f'for_{task}'])
    else: # python convert_to_h5.py for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        convert_tiffs_to_h5(task)
