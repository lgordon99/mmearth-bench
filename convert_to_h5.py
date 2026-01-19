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
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
pixel_level_modalities = ['Sentinel2',
                          'Sentinel1',
                          'ASTER_GDEM',
                          'ETH_GCH',
                          'DynamicWorld',
                          'ESA_WorldCover',
                          'MSK_CLDPRB',
                          'S2CLOUDLESS',
                          'SCL']
tile_level_modalities = ['precipitation',
                         'temperature',
                         'geolocation',
                         'geolocation_encoding',
                         'month_encoding',
                         'biome',
                         'ecoregion',
                         'MSK_CLDPRB_CLOUDY_PIXEL_FRACTION',
                         'S2CLOUDLESS_CLOUDY_PIXEL_FRACTION',
                         'SCL_NO_DATA_PIXEL_FRACTION']

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
                data['sentinel2_system_index'].append(tags['sentinel2_system_index'])
                data['crs'].append(tiff.crs.to_string())
                data['transform'].append([i for i in tiff.transform])
                data['missing_modalities'].append(tags['missing_modalities'])

        for key, value in data.items():
            h5_file.create_dataset(key, data=value, compression='gzip', compression_opts=9)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_h5(task):
    with open(f'{data_dir_path}/{task}/output-files/check_h5.out', 'w') as out_file:
        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            tile_data = {modality: h5_file[modality][:] for modality in pixel_level_modalities + tile_level_modalities}
            out_file.write(f'{len(tile_data[list(tile_data.keys())[0]])} tiles made\n')

            for key in h5_file.keys():
                value = h5_file[key][:]
                out_file.write(f'{key}: {value.dtype}\n')

                if key in pixel_level_modalities or key in tile_level_modalities or 'soil' in key:
                    if key in no_data_values:
                        value = np.ma.masked_equal(value, no_data_values[key])
                        out_file.write(f'NaN pixels = {(value.mask.sum() / value.size) * 100}%\n')

                    if key == 'ASTER_GDEM':
                        out_file.write('Elevation band\n')
                        out_file.write(f'Min: {np.min(value[:, 0])}, Max: {np.max(value[:, 0])}\n')
                        out_file.write('Slope band\n')
                        out_file.write(f'Min: {np.min(value[:, 1])}, Max: {np.max(value[:, 1])}\n')
                    elif key == 'ETH_GCH':
                        out_file.write('Height band\n')
                        out_file.write(f'Min: {np.min(value[:, 0])}, Max: {np.max(value[:, 0])}\n')
                        out_file.write('Uncertainty band\n')
                        out_file.write(f'Min: {np.min(value[:, 1])}, Max: {np.max(value[:, 1])}\n')
                    else:
                        out_file.write(f'Min: {np.min(value)}, Max: {np.max(value)}\n')

                    if key == 'Sentinel2':
                        assert np.min(value) >= 0
                        assert np.max(value) < no_data_values[key]
                    elif key == 'Sentinel1' or key == 'ASTER_GDEM':
                        assert np.min(value) > no_data_values[key]
                    elif key == 'ETH_GCH':
                        assert np.max(value) < no_data_values[key]
                    elif key in ['DynamicWorld', 'ESA_WorldCover', 'biome', 'ecoregion']:
                        assert np.min(value) >= 0
                        assert np.max(value) < no_data_values[key]
                    elif key == 'precipitation':
                        assert np.min(value) >= 0
                    elif key == 'geolocation':
                        assert np.min(value) >= -180
                        assert np.max(value) <= 180
                    elif 'encoding' in key:
                        assert np.min(value) >= -1
                        assert np.max(value) <= 1

                out_file.write('\n')

            out_file.write(f'Sentinel-2 date: {h5_file["sentinel2_date"].asstr()[...]}\n')
            out_file.write(f'Sentinel-2 system index: {h5_file["sentinel2_system_index"].asstr()[...]}\n')
            out_file.write(f'CRS: {h5_file["crs"].asstr()[...]}\n')
            out_file.write(f'ID: {h5_file["id"][:10]}\n')
            out_file.write(f'Missing modalities: {[json.loads(lst) for lst in h5_file["missing_modalities"].asstr()[...]][:10]}\n')

            for modality in ['ASTER_GDEM', 'ETH_GCH']:
                modality_data = tile_data[modality]
                modality_data_reshaped = modality_data.reshape(len(modality_data), 2, 16384)
                nan_indices = np.argwhere(modality_data_reshaped == no_data_values[modality])

                for numbers in nan_indices:
                    x, y, z = numbers

                    if y == 0:
                        assert modality_data_reshaped[x][1][z] == no_data_values[modality] # slope (uncertainty) band should be NaN wherever the elevation (height) band is NaN

def plot_missing_modalities(task):
    with open(f'{data_dir_path}/{task}/output-files/plot_missing_modalities.out', 'w') as out_file:
        original_num_tiles = len(utils.read_geojson(f'{data_dir_path}/{task}/{task}_points.geojson')['features'])
        out_file.write(f'Original number of tiles = {original_num_tiles}\n')

        with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
            missing_modalities = [json.loads(lst) for lst in h5_file['missing_modalities'].asstr()[...]]

        out_file.write(f'Final number of tiles = {len(missing_modalities)}\n')

    assert len(missing_modalities) == len(os.listdir(f'{data_dir_path}/{task}/tiffs'))

    missing_modality_counts = dict(Counter(item for sublist in missing_modalities for item in sublist))
    missing_modality_counts['sentinel2'] = original_num_tiles - len(missing_modalities)
    modality_order = ['sentinel2', 'sentinel1', 'aster_gdem', 'eth_gch', 'dynamic_world', 'esa_worldcover', 'precipitation', 'temperature', 'biome', 'ecoregion']
    missing_modality_counts = {modality: missing_modality_counts.get(modality, 0) for modality in modality_order}

    axes_pos = [0.14, 0.2, 0.8, 0.7] # left, bottom, width, height

    if task == 'species':
        plt.figure(figsize=(6, 3.5), dpi=300)
    else:
        plt.figure(figsize=(6, 2.5), dpi=300)

    ax = plt.gca()
    ax.set_position(axes_pos)
    plt.bar(['Sentinel-2', 'Sentinel-1', 'ASTER GDEM', 'ETH GCH', 'Dynamic World', 'ESA WorldCover', 'Precipitation', 'Temperature', 'Biome', 'Ecoregion'], missing_modality_counts.values())

    if task == 'species':
        plt.xlabel('Modality', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
    else:
        plt.tick_params(labelbottom=False)

    plt.ylabel('Tile count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{data_dir_path}/{task}/{task}_missing_modality_counts.pdf')
    plt.close()

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
        plt.xticks(indices, species, rotation=90, fontsize=12)
        plt.xlabel('Species', fontsize=12)
        plt.ylabel('Number of tiles', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{data_dir_path}/species/tiles_per_species.pdf')

        ax = plt.gca()
        fig = plt.gcf()

        # transparent background
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')

        # make all text/ticks white
        ax.tick_params(axis='both', colors='white')  # tick marks + tick label color
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')

        # remove the axes "box" (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # remove bar borders (if any backend adds them)
        for patch in ax.patches:
            patch.set_edgecolor('none')
            patch.set_linewidth(0)

        plt.savefig(f'{data_dir_path}/species/tiles_per_species.svg', format='svg', transparent=True, bbox_inches='tight', pad_inches=0)

        out_file.write(f'Max number of tiles for a species: {max(counts)}\n')
        out_file.write(f'Min number of tiles for a species: {min(counts)}\n')

        # species richness per tile
        plt.figure(dpi=300)
        num_species_per_tile = [len(tile) for tile in tiles]
        tile_counts = Counter(num_species_per_tile) # counts the number of tiles for each number of species
        num_species = sorted(tile_counts.keys()) # sorts the number of species in ascending order
        num_tiles = [tile_counts[n] for n in num_species] # counts the number of tiles for each number of species

        plt.bar(num_species, num_tiles, edgecolor='black')
        plt.xlabel('Number of species', fontsize=12)
        plt.ylabel('Number of tiles', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{data_dir_path}/species/tile_species_counts.pdf')

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
