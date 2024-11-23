'''
get_tile_data.py by Lucia Gordon and Vishal Nedungadi
'''

# imports
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ee_data import EEData
from sys import argv
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

def get_biomass_dates_points(tile):
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
    print(f'{points.size().getInfo()} points')
    leaf_on_off = np.array([points.aggregate_array('leaf_on_doy').getInfo(), points.aggregate_array('leaf_off_doy').getInfo()]).T # get pairs of leaf on and off days for each point
    leaf_on_off_unique = list(map(list, set(map(tuple, leaf_on_off)))) # get unique pairs
    leaf_on_off_dates = [[pd.to_datetime(pair[0], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d'), pd.to_datetime(pair[1], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d')] for pair in leaf_on_off_unique]
    # TODO: extract agbd value only and call it something like value and set the property
    return leaf_on_off_dates, points

def get_soil_points(property_, tile):
    dataframe = pd.read_csv(f'{data_dir_path}/{property_}.csv')
    points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude])).set({'value': measurement.value_avg, 'pos_uncertainty': measurement.positional_uncertainty}) for measurement in dataframe.itertuples()]).filterBounds(tile['geometry'])
    # TODO: extract value and set it as a property in the feature to be read in in ee_data
    # return points
    print(points.size().getInfo())
    print(points.aggregate_array('value').getInfo())
    print(points.aggregate_array('pos_uncertainty').getInfo())

def get_modalities_for_biomass():
    task = 'biomass'
    os.makedirs(f'{task}/pixel_level_data', exist_ok=True)
    gj = utils.read_geojson(f'{task}/{task}_tiles.geojson') # reading the GeoJSON file
    tile_image_level_data = {}
    start_tile = 0
    end_tile = len(gj['features'])
    tiles_made = 0

    for tile_index in range(start_tile, end_tile):
        print(f'Processing tile {tile_index+1}/{end_tile}')
        tile = gj['features'][tile_index]
        leaf_on_off_dates, points = get_biomass_dates_points(tile)

        if len(leaf_on_off_dates) > 0:
            ee_data = EEData(tile, task, leaf_on_off_dates, points)

            if not ee_data.no_data:
                tiles_made += 1
                tile_image_level_data[tile['id']] = {'biome': ee_data.biome,
                                                     'ecoregion': ee_data.ecoregion,
                                                     'era5': ee_data.era5_data,
                                                     's2_date': ee_data.s2_date,
                                                     'geolocation_encoding': ee_data.geolocation_encoding,
                                                     'month_encoding': ee_data.month_encoding,
                                                     'crs': ee_data.crs,
                                                     'lat': ee_data.lat,
                                                     'lon': ee_data.lon}

    with open(f'tiles/{task}/tile_image_level_data.json', 'w') as file:
        json.dump(tile_image_level_data, file, indent=4)

    print(f'{tiles_made} tiles made')

def get_modalities_for_species():
    task = 'species'
    os.makedirs(f'{task}/pixel_level_data', exist_ok=True)
    gj = utils.read_geojson(f'{task}/tiles/{task}_tiles.geojson') # reading the GeoJSON file
    tile_image_level_data = {}
    start_tile = 8714
    end_tile = len(gj['features'])
    tiles_made = 0

    for tile_index in range(start_tile, end_tile):
        print(f'Processing tile {tile_index}/{end_tile-1}')
        tile = gj['features'][tile_index]
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

        ee_data = EEData(tile, task, dates)

        if not ee_data.no_data:
            tiles_made += 1
            tile_image_level_data[tile['id']] = {'biome': ee_data.biome,
                                                    'ecoregion': ee_data.ecoregion,
                                                    'era5': ee_data.era5_data,
                                                    's2_date': ee_data.s2_date,
                                                    'geolocation_encoding': ee_data.geolocation_encoding,
                                                    'month_encoding': ee_data.month_encoding,
                                                    'crs': ee_data.crs,
                                                    'lat': ee_data.lat,
                                                    'lon': ee_data.lon}

    with open(f'tiles/{task}/tile_image_level_data.json', 'w') as file:
        json.dump(tile_image_level_data, file, indent=4)

    print(f'{tiles_made} tiles made')

def get_modalities_for_soil(task):
    start_time = time.time()

    os.makedirs(f'{task}/pixel_level_data', exist_ok=True)
    gj = utils.read_geojson(f'{task}/{task}_tiles.geojson') # reading the GeoJSON file
    tile_image_level_data = {}
    start_tile = 0
    end_tile = len(gj['features'])
    tiles_made = 0

    for tile_index in range(start_tile, end_tile):
        print(f'Processing tile {tile_index+1}/{end_tile}')
        tile = gj['features'][tile_index]
        tile_center_latitude = utils.get_rectangle_center(tile['geometry']['coordinates'][0])[1]

        if tile_center_latitude > 0:
            dates = [[f'{year}-{str(5).zfill(2)}-01', f'{year}-{str(9).zfill(2)}-{get_last_day_of_month(9)}']]
        elif tile_center_latitude < 0:
            dates = [[f'{year}-{str(11).zfill(2)}-01', f'{year}-{str(12).zfill(2)}-{get_last_day_of_month(12)}'], [f'{year}-{str(1).zfill(2)}-01', f'{year}-{str(3).zfill(2)}-{get_last_day_of_month(3)}']]

        ee_data = EEData(tile, task, dates)

        if not ee_data.no_data:
            tiles_made += 1
            tile_image_level_data[tile['id']] = {'biome': ee_data.biome,
                                                 'ecoregion': ee_data.ecoregion,
                                                 'era5': ee_data.era5_data,
                                                 's2_date': ee_data.s2_date,
                                                 'lat': ee_data.lat,
                                                 'lon': ee_data.lon,
                                                 'month_encoding': ee_data.month_encoding,
                                                 'crs': ee_data.crs,
                                                 'lat': ee_data.lat,
                                                 'lon': ee_data.lon,
                                                 task: tile['properties']['value']}

    with open(f'{task}/{task}_tile_image_level_data.json', 'w') as file:
        json.dump(tile_image_level_data, file, indent=4)

    print(f'{tiles_made} tiles made')
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

if __name__ == '__main__':
    # get_modalities_for_biomass()
    # get_modalities_for_species()
    # get_modalities_for_soil(task='soil_nitrogen')
    # get_modalities_for_soil(task='soil_organic_carbon')
    # get_modalities_for_soil(task='soil_pH')
    # exit()
    if argv[1] == 'biomass':
        subprocess.run(['sbatch', '-t', '1-00:00:00', '-p', partitions, '-o', 'bash-outputs/biomass_mmearth_modalities.out', '-e', 'bash-errors/biomass_mmearth_modalities.err', 'job.sh', env_path, 'get_tile_data.py', 'get_modalities_for_biomass'])
    elif argv[1] == 'get_modalities_for_biomass':
        get_modalities_for_biomass()
    elif argv[1] == 'species':
        subprocess.run(['sbatch', '-t', '1-00:00:00', '-p', partitions, '--job-name', 'species_mmearth_modalities', '-o', 'bash-outputs/species_mmearth_modalities.out', '-e', 'bash-errors/species_mmearth_modalities.err', 'job.sh', env_path, 'get_tile_data.py', 'get_modalities_for_species'])
    elif argv[1] == 'get_modalities_for_species':
        get_modalities_for_species()
    elif argv[1] == 'soil_nitrogen':
        subprocess.run(['sbatch', '-t', '1-00:00:00', '-p', partitions, '-o', 'bash-outputs/soil_nitrogen_mmearth_modalities.out', '-e', 'bash-errors/soil_nitrogen_mmearth_modalities.err', 'job.sh', env_path, 'get_tile_data.py', 'get_modalities_for_soil_nitrogen'])
    elif argv[1] == 'get_modalities_for_soil_nitrogen':
        get_modalities_for_soil(task='soil_nitrogen')
