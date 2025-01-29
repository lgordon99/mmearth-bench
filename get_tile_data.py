'''
get_tile_data.py
'''

# ============================================== IMPORTS ============================================== #

from ee_data import EEData
from sys import argv
import ee
import matplotlib.pyplot as plt
import os
import subprocess
import time
import utils
import yaml

# ============================================== GLOBAL VARIABLES ============================================== #

ee.Initialize(project='mmearth-bench') # initializes EE with our project
year = '2020'
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)

# ============================================== FUNCTIONS ============================================== #

def get_modalities(task):
    start_time = time.time()
    os.makedirs(f'{data_dir_path}/{task}/data', exist_ok=True)
    points = utils.read_geojson(f'{task}/points/{task}_points.geojson') # reading the GeoJSON file
    end_point = len(points['features'])
    tiles_made = 0
    tile_missing_modalities_yml_path = f'{task}/{task}_missing_modalities.yml'

    if os.path.exists(tile_missing_modalities_yml_path): # if there is some data saved
        tile_missing_modalities = utils.read_yaml(tile_missing_modalities_yml_path)
        start_point = next(reversed(tile_missing_modalities)) + 1
    else:
        tile_missing_modalities = {}
        start_point = 0

    for point_id in range(start_point, end_point):
        print(f'Processing tile {point_id}/{end_point-1}')
        point = points['features'][point_id]
        ee_data = EEData(point, task, data_dir_path)
        tile_missing_modalities[point_id] = ee_data.missing_modalities

        with open(tile_missing_modalities_yml_path, 'w') as file:
            yaml.dump(tile_missing_modalities, file, default_flow_style=False)

        if not ee_data.no_data:
            tiles_made += 1

    print(f'{tiles_made} tiles made')
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_complete(task):
    tile_count = len(utils.read_geojson(f'{task}/points/{task}_points.geojson')['features'])
    tile_missing_modalities = utils.read_yaml(f'{task}/{task}_missing_modalities.yml')
    print(f'All tiles made for {task}: {tile_count == next(reversed(tile_missing_modalities)) + 1}')

def plot_missing_modalities(task):
    print(task)

    tiles = utils.read_geojson(f'{task}/points/{task}_points.geojson')['features']
    print(f'Number of tiles = {len(tiles)}')

    tile_missing_modalities = utils.read_yaml(f'{task}/{task}_missing_modalities.yml')
    modalities = ['sentinel2', 'sentinel1', 'aster', 'canopy_height_eth', 'dynamic_world', 'esa_worldcover', 'era5', 'biome/ecoregion']
    missing_modality_counts = {modality: 0 for modality in modalities}

    for tile_id in range(len(tiles)):
        missing_modalities = tile_missing_modalities[tile_id]

        if missing_modalities:
            for modality in missing_modalities:
                missing_modality_counts[modality] += 1

    num_failed_tiles = sum([missing_modality_counts[modality] for modality in modalities])
    print(f'Number of tiles after getting modalities = {len(tiles) - num_failed_tiles}')

    plt.figure(dpi=300)
    plt.bar(missing_modality_counts.keys(), missing_modality_counts.values())
    plt.title(f'{task}: Missing Modality Counts', fontsize=14)
    plt.xlabel('Modalities', fontsize=12)
    plt.ylabel('Tile count', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{task}/figures/{task}_missing_modality_counts.png')

# ============================================== RUN ============================================== #

if __name__ == '__main__':
    if 'check_complete' in argv[1]: # python get_tile_data.py check_complete TASK
        check_complete(argv[2])
    elif 'plot_missing_modalities' in argv[1]: # python get_tile_data.py plot_missing_modalities TASK
        plot_missing_modalities(argv[2])
    elif 'for' not in argv[1]: # python get_tile_data.py TASK
        subprocess.run(['sbatch', '-t', '5-00:00:00', '-p', partitions, '--mem', '500M', '--job-name', f'{argv[1]}_mmearth_modalities', '-o', f'bash-outputs/{argv[1]}_mmearth_modalities.out', '-e', f'bash-errors/{argv[1]}_mmearth_modalities.err', 'job.sh', env_path, 'get_tile_data.py', f'get_modalities_for_{argv[1]}'])
    elif 'for' in argv[1]: # python get_tile_data.py get_modalities_for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        get_modalities(task)
