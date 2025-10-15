# ============================================== IMPORTS ============================================== #

from collections import Counter, defaultdict
from sys import argv
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import subprocess
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'DynamicWorld', 'ESA_WorldCover', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL']
tile_level_modalities = ['precipitation', 'temperature', 'geolocation', 'geolocation_encoding', 'month_encoding', 'biome', 'ecoregion', 'MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', 'S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', 'SCL_NO_DATA_PIXEL_FRACTION']

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
                elif task == 'species':
                    data[task].append(tags[task])

                # additional tile data
                data['id'].append(get_tile_id(tiff_filename))
                data['sentinel2_date'].append(tags['sentinel2_date'])
                data['crs'].append(tiff.crs.to_string())
                data['transform'].append([i for i in tiff.transform])
                data['missing_modalities'].append(tags['missing_modalities'])

        for key, value in data.items():
            h5_file.create_dataset(key, data=value, compression='gzip', compression_opts=9)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_h5(task):
    with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
        tile_data = {modality: h5_file[modality][:] for modality in pixel_level_modalities + tile_level_modalities}
        aster_dem = tile_data['AsterDEM']
        print(f'{len(aster_dem)} tiles made')
        print(h5_file.keys())

        for key in h5_file.keys():
            value = h5_file[key][:]

            print(f'{key}: {value.dtype}')

            if key in pixel_level_modalities or key in tile_level_modalities or key == 'soil_nitrogen' or key == 'id':
                print(f'Min: {np.min(value)}, Max: {np.max(value)}')

                if key == 'Sentinel2':
                    assert np.min(value) >= 0
                    assert np.max(value) <= 65535
                elif key == 'Sentinel1' or key == 'AsterDEM' or key == 'precipitation' or key == 'temperature' or key == 'geolocation' or key == 'month':
                    assert np.min(value) >= -9999
                elif key == 'ETH_GCH':
                    assert np.max(value) <= 255
                elif key == 'DynamicWorld':
                    assert np.min(value) >= 0
                    assert np.max(value) <= 9
                elif key == 'ESA_WorldCover':
                    assert np.min(value) >= 0
                    assert np.max(value) <= 11
                elif key == 'biome':
                    assert np.min(value) >= 0
                    assert np.max(value) <= 14
                elif key == 'ecoregion':
                    assert np.min(value) >= 0
                    assert np.max(value) <= 846

        print(h5_file['sentinel2_date'].asstr()[...])
        print(h5_file['crs'].asstr()[...])
        print(h5_file['id'][:5])
        print([json.loads(lst) for lst in h5_file['missing_modalities'].asstr()[...]][:5])

        aster_dem_reshaped = aster_dem.reshape(len(aster_dem), 2, 16384)
        assert np.count_nonzero(aster_dem_reshaped[:, 0] == 0) == 0 # should be no zeros in the elevation band
        nan_indices = np.argwhere(aster_dem_reshaped == -9999)

        for numbers in nan_indices:
            x, y, z = numbers

            if y == 0:
                assert aster_dem_reshaped[x][1][z] == -9999 # slope band should be NaN wherever the elevation band is NaN

        eth_gch = tile_data['ETH_GCH']
        eth_gch_reshaped = eth_gch.reshape(len(eth_gch), 2, 16384)
        nan_indices = np.argwhere(eth_gch_reshaped == 255)

        for numbers in nan_indices:
            x, y, z = numbers

            if y == 0:
                assert eth_gch_reshaped[x][1][z] == 255 # uncertainty band should be NaN wherever the height band is NaN

def plot_missing_modalities(task):
    print(task)
    original_num_tiles = len(utils.read_geojson(f'{data_dir_path}/{task}/{task}_points.geojson')['features'])
    print(f'Original number of tiles = {original_num_tiles}')

    with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
        missing_modalities = [json.loads(lst) for lst in h5_file['missing_modalities'].asstr()[...]]

    print(f'Final number of tiles = {len(missing_modalities)}')
    assert len(missing_modalities) == len(os.listdir(f'{data_dir_path}/{task}/tiffs'))

    missing_modality_counts = dict(Counter(item for sublist in missing_modalities for item in sublist))
    missing_modality_counts['sentinel2'] = original_num_tiles - len(missing_modalities)
    modality_order = ['sentinel2', 'sentinel1', 'aster', 'eth_gch', 'dynamic_world', 'esa_worldcover', 'precipitation', 'temperature', 'biome', 'ecoregion']
    missing_modality_counts = {modality: missing_modality_counts.get(modality, 0) for modality in modality_order}

    plt.figure(dpi=300)
    plt.bar(['Sentinel-2', 'Sentinel-1', 'AsterDEM', 'ETH GCH', 'Dynamic World', 'ESA WorldCover', 'Precipitation', 'Temperature', 'Biome', 'Ecoregion'], missing_modality_counts.values())
    plt.title(f'{task.capitalize().replace("_", " ").replace("ph", "pH")} missing modality counts', fontsize=14)
    plt.xlabel('Modality', fontsize=12)
    plt.ylabel('Tile count', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{data_dir_path}/{task}/{task}_missing_modality_counts.png')

def plot_species_statistics():
    with open(f'{data_dir_path}/species/output-files/check_species_statistics.out', 'w') as out_file:
        with h5py.File(f'{data_dir_path}/species/species.h5', 'r') as h5_file:
            tiles = [json.loads(lst) for lst in h5_file['species'].asstr()[...]]

        out_file.write(f'Total tiles: {len(tiles)}\n')

        # plot number of tiles per species
        species_counts = {}

        for tile in tiles:
            for species in tile:
                species_counts[species] = species_counts.get(species, 0) + 1

        species_counts = dict(sorted(species_counts.items(), key=lambda item: item[1], reverse=True)) # sorts by value in descending order
        species = list(species_counts.keys())
        counts = list(species_counts.values())
        indices = np.arange(len(species))
        plt.figure(dpi=300, figsize=(20, 5))
        plt.bar(indices, counts)
        plt.margins(x=1e-2)
        plt.xticks(indices, species, rotation=90)
        plt.xlabel('Species')
        plt.ylabel('Number of tiles')
        plt.title('Number of tiles per species')
        plt.tight_layout()
        plt.savefig(f'{data_dir_path}/species/tiles_per_species.png')

        out_file.write(f'Max number of tiles for a species: {max(counts)}\n')
        out_file.write(f'Min number of tiles for a species: {min(counts)}\n')

        # histogram of number of species per tile
        plt.figure(dpi=300)
        bin_size = 1000
        bins = np.arange(0, max(counts) + bin_size, bin_size)
        tick_interval = 5000

        plt.hist(counts, bins=bins, edgecolor='black')
        plt.xlabel('Number of tiles')
        plt.ylabel('Number of species')
        plt.xticks(np.arange(0, max(counts) + tick_interval, tick_interval))
        plt.tight_layout()
        plt.savefig(f'{data_dir_path}/species/tile_species_counts.png')

        with open(f'{data_dir_path}/species/species_labels.json', 'w') as file:
            json.dump({name: i for i, name in enumerate(species)}, file, indent=4)

if __name__ == '__main__':
    if 'check_h5' in argv[1]: # python convert_to_h5.py check_h5 TASK
        check_h5(argv[2])
    elif 'plot_missing_modalities' in argv[1]: # python convert_to_h5.py plot_missing_modalities TASK
        plot_missing_modalities(argv[2])
    elif 'plot_species_statistics' in argv[1]: # python convert_to_h5.py plot_species_statistics
        plot_species_statistics()
    elif 'for' not in argv[1]: # python convert_to_h5.py TASK
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        task = argv[1]

        if task == 'biomass':
            mem = 300
        elif 'soil' in task:
            mem = 60
        elif task == 'species':
            mem = 100

        subprocess.run(['sbatch', '-t', '15:00:00', '-p', partitions, '--mem', f'{mem}G', '--job-name', f'{task}_convert_to_h5', '-o', f'{data_dir_path}/{task}/output-files/{task}_convert_to_h5.out', '--account', 'davies_lab', 'job.sh', env_path, 'convert_to_h5.py', f'for_{task}'])
    else: # python convert_to_h5.py for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        convert_tiffs_to_h5(task)
