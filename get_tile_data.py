'''
get_tile_data.py by Lucia Gordon and Vishal Nedungadi
'''

# imports
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ee_data import EEData
from sys import argv
# import csv
import ee
# import geojson
# import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import time
import utils
import yaml

ee.Initialize(project='mmearth-bench') # initializes EE with our project
year = '2020'
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)

# def get_last_day_of_month(month):
#     return (datetime(int(year), month, 1) + relativedelta(months=1, days=-1)).day

# def get_gedi_points(tile):
#     collection_names = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
#                           .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"') # get feature collections with features in the selected year
#                           .filterBounds(tile['properties']['outer_tile']) # get feature collections that have features within the tile
#                           .aggregate_array('table_id') # extract the IDs of the feature collections
#                           .getInfo()) # list of names of the feature collections
#     quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
#     gedi_points = (ee.FeatureCollection([result for _, result in ((collection_name, utils.get_asset_if_valid(ee.FeatureCollection(collection_name))) for collection_name in collection_names) if result is not None])
#                      .flatten() # merge all the feature collections into one
#                      .filterBounds(tile['properties']['outer_tile']) # collection of the features that are within the tile
#                      .filter(quality_filter) # apply the quality filter
#                      .map(lambda point: point.set('off_after_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
#                      .filter(ee.Filter.gt('off_after_on', 0))) # filtering by GEDI points with leaf on before leaf off

#     return gedi_points

# def get_dates(task, point):
#     if task == 'biomass':
#         points = get_gedi_points(point)
#         leaf_on_off = np.array([points.aggregate_array('leaf_on_doy').getInfo(), points.aggregate_array('leaf_off_doy').getInfo()]).T # get pairs of leaf on and off days for each point
#         leaf_on_off_unique = list(map(list, set(map(tuple, leaf_on_off)))) # get unique pairs
#         leaf_on_off_dates = [[pd.to_datetime(pair[0], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d'), pd.to_datetime(pair[1], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d')] for pair in leaf_on_off_unique]

#         return leaf_on_off_dates

#     elif task == 'species':
#         month = point['properties']['month']

#         if month > 1 and month < 12:
#             start_month = month - 1
#             end_month = month + 1
#             dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]
#         elif month == 1:
#             start_month = 12
#             end_month = 2
#             dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(start_month).zfill(2)}-31'], [f'{year}-{str(month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]
#         elif month == 12:
#             start_month = 11
#             end_month = 1
#             dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(month).zfill(2)}-31'], [f'{year}-{str(end_month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]

#         return dates

#     elif 'soil' in task:
#         point_latitude = point['geometry']['coordinates'][1]

#         if point_latitude > 0:
#             dates = [[f'{year}-{str(5).zfill(2)}-01', f'{year}-{str(9).zfill(2)}-{get_last_day_of_month(9)}']]
#         elif point_latitude < 0:
#             dates = [[f'{year}-{str(11).zfill(2)}-01', f'{year}-{str(12).zfill(2)}-{get_last_day_of_month(12)}'], [f'{year}-{str(1).zfill(2)}-01', f'{year}-{str(3).zfill(2)}-{get_last_day_of_month(3)}']]

#         return dates

# def get_task_values(task, point):
#     if task == 'biomass':
#         return get_gedi_points(point)

#     elif task == 'species':
#         species = point['properties']['species']
#         main_species = point['properties']['main_species']

#         return {'species': species, 'main_species': main_species}

#     elif 'soil' in task:
#         return {task: point['properties']['value']}

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

def get_modalities(task):
    start_time = time.time()
    os.makedirs(f'{task}/data', exist_ok=True)
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
        ee_data = EEData(point, task)
        tile_missing_modalities[point_id] = ee_data.missing_modalities

        with open(tile_missing_modalities_yml_path, 'w') as file:
            yaml.dump(tile_missing_modalities, file, default_flow_style=False)

        if not ee_data.no_data:
            tiles_made += 1

    print(f'{tiles_made} tiles made')
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

if __name__ == '__main__':
    if 'plot_missing_modalities' in argv[1]:
        plot_missing_modalities(argv[2])
    elif 'for' not in argv[1]:
        subprocess.run(['sbatch', '-t', '3-00:00:00', '-p', partitions, '--mem', '500M', '--job-name', f'{argv[1]}_mmearth_modalities', '-o', f'bash-outputs/{argv[1]}_mmearth_modalities.out', '-e', f'bash-errors/{argv[1]}_mmearth_modalities.err', 'job.sh', env_path, 'get_tile_data.py', f'get_modalities_for_{argv[1]}'])
    elif 'for' in argv[1]:
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        get_modalities(task)
