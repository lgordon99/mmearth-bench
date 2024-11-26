'''
get_tile_data.py by Lucia Gordon and Vishal Nedungadi
'''

# imports
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ee_data import EEData
from sys import argv
import csv
import ee
import geojson
import json
import numpy as np
import os
import pandas as pd
import subprocess
import time
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project
year = '2020'
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)

def get_last_day_of_month(month):
    return (datetime(int(year), month, 1) + relativedelta(months=1, days=-1)).day

def get_biomass_points(tile):
    collection_names = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                            .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"') # get feature collections with features in the selected year
                            .filterBounds(tile['geometry']) # get feature collections that have features within the tile
                            .aggregate_array('table_id') # extract the IDs of the feature collections
                            .getInfo()) # list of names of the feature collections
    quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
    points = (ee.FeatureCollection([ee.FeatureCollection(name) for name in collection_names])
                .flatten() # merge all the feature collections into one
                .filterBounds(tile['geometry']) # collection of the features that are within the tile
                .filter(quality_filter) # apply the quality filter
                .map(lambda point: point.set('off_after_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
                .filter(ee.Filter.gt('off_after_on', 0))) # filtering by points with leaf on before leaf off

    return points

def get_dates(task, tile):
    if task == 'biomass':
        points = get_biomass_points(tile)
        leaf_on_off = np.array([points.aggregate_array('leaf_on_doy').getInfo(), points.aggregate_array('leaf_off_doy').getInfo()]).T # get pairs of leaf on and off days for each point
        leaf_on_off_unique = list(map(list, set(map(tuple, leaf_on_off)))) # get unique pairs
        leaf_on_off_dates = [[pd.to_datetime(pair[0], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d'), pd.to_datetime(pair[1], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d')] for pair in leaf_on_off_unique]

        return leaf_on_off_dates
    elif task == 'species':
        month = tile['properties']['month']

        if month > 1 and month < 12:
            start_month = month - 1
            end_month = month + 1
            dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]
        elif month == 1:
            start_month = 12
            end_month = 2
            dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(start_month).zfill(2)}-31'], [f'{year}-{str(month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]
        elif month == 12:
            start_month = 11
            end_month = 1
            dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(month).zfill(2)}-31'], [f'{year}-{str(end_month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]

        return dates
    elif 'soil' in task:
        tile_center_latitude = utils.get_rectangle_center(tile['geometry']['coordinates'][0])[1]

        if tile_center_latitude > 0:
            dates = [[f'{year}-{str(5).zfill(2)}-01', f'{year}-{str(9).zfill(2)}-{get_last_day_of_month(9)}']]
        elif tile_center_latitude < 0:
            dates = [[f'{year}-{str(11).zfill(2)}-01', f'{year}-{str(12).zfill(2)}-{get_last_day_of_month(12)}'], [f'{year}-{str(1).zfill(2)}-01', f'{year}-{str(3).zfill(2)}-{get_last_day_of_month(3)}']]

        return dates

def get_task_values(task, tile):
    if task == 'biomass':
        return get_biomass_points(tile)
    elif task == 'species':
        species = tile['properties']['species']
        main_species = tile['properties']['main_species']

        return {'species': species, 'main_species': main_species}
    elif 'soil' in task:
        return {task: tile['properties']['value']}

def get_modalities(task):
    start_time = time.time()
    os.makedirs(f'{task}/data', exist_ok=True)
    gj = utils.read_geojson(f'{task}/tiles/{task}_tiles.geojson') # reading the GeoJSON file
    error_file_path = f'bash-errors/{task}_mmearth_modalities.err'
    start_tile = 0

    if os.path.exists(error_file_path): # if error file exists
        with open(error_file_path, 'r') as error_file:
            error_content = error_file.read().replace('*** Earth Engine *** Share your feedback by taking our Annual Developer Satisfaction Survey: https://google.qualtrics.com/jfe/form/SV_0JLhFqfSY1uiEaW?source=Init\n', '')

            if len(error_content) > 0: # if there is an error
                with open(f'bash-outputs/{task}_mmearth_modalities.out', 'r') as out_file:
                    out_file = out_file.read()
                    start_tile = int(content.split('\n')[-2].split('/')[0].split(' ')[-1])

    end_tile = len(gj['features'])
    tiles_made = 0
    tile_missing_modalities = []

    for tile_index in range(start_tile, end_tile):
        print(f'Processing tile {tile_index}/{end_tile-1}')
        tile = gj['features'][tile_index]
        dates = get_dates(task, tile)
        task_values = get_task_values(task, tile)

        if len(dates) > 0:
            ee_data = EEData(tile, task, dates, task_values)

            if ee_data.no_data:
                tile_missing_modalities.append([tile_index, ee_data.modality_returned_false])

                with open(f'{task}/{task}_missing_modalities.csv', 'w', newline='') as file:
                    writer = csv.writer(file)

                    for row in tile_missing_modalities:
                        writer.writerow(row)
            else:
                tiles_made += 1

    print(f'{tiles_made} tiles made')
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

if __name__ == '__main__':
    if 'for' not in argv[1]:
        subprocess.run(['sbatch', '-t', '2-00:00:00', '-p', partitions, '--job-name', f'{argv[1]}_mmearth_modalities', '-o', f'bash-outputs/{argv[1]}_mmearth_modalities.out', '-e', f'bash-errors/{argv[1]}_mmearth_modalities.err', 'job.sh', env_path, 'get_tile_data.py', f'get_modalities_for_{argv[1]}'])
    else:
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        get_modalities(task)
