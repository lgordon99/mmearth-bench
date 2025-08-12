'''
generate_biomass_points.py by Lucia Gordon
A script to create a GeoJSON with biomass points balanced across biomes
'''

# ============================================== IMPORTS ============================================== #

from google.cloud import storage
from shapely import bounds
from sys import argv
import builtins
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import functools
import geemap
import geopandas as gpd
import getpass
import json
import math
import matplotlib.pyplot as plt
import os
import random
import subprocess
import sys
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

random.seed(42)
print = functools.partial(builtins.print, flush=True)
ee.Initialize(project='mmearth-bench') # initializes EE with our project
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
client = storage.Client()
bucket = client.bucket('biomass_ecoregion_data')
OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']

# ============================================== FUNCTIONS ============================================== #

def get_ecoregion_collection():
    return ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filter(ee.Filter.neq('BIOME_NAME', 'N/A')) # dataset with the extents of the 846 terrestrial ecoregions

def save_biome_ecoregion_labels():
    ecoregion_collection = get_ecoregion_collection()
    biomes = sorted(set(ecoregion_collection.aggregate_array('BIOME_NAME').getInfo())) # list of 14 biomes
    ecoregions = sorted(set(ecoregion_collection.aggregate_array('ECO_NAME').getInfo())) # list of 846 ecoregions

    # assign each biome and ecoregion an integer label
    for item in ['biome', 'ecoregion']:
        with open(f'{data_dir_path}/{item}_labels.json', 'w') as file:
            json.dump({name: i for i, name in enumerate(locals()[f'{item}s'])}, file, indent=4)

def get_ecoregions_in_gedi_collection():
    gedi_latitude_range = [-51.6, 51.6] # GEDI covers the latitude band between 51.6 degees N and S
    gedi_range_polygon = ee.Geometry.Polygon(coords=[[-180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[1]],
                                                    [-180, gedi_latitude_range[1]]],
                                             proj=None,
                                             geodesic=False) # polygon covering GEDI range
    ecoregions_in_gedi_collection = (get_ecoregion_collection()
                                     .filterBounds(gedi_range_polygon) # excludes ecoregions outside of the GEDI range, leaving 774
                                     .map(lambda feature: feature.setGeometry(feature.geometry().intersection(gedi_range_polygon, maxError=1)))) # crops ecoregions to the GEDI range

    return ecoregions_in_gedi_collection

def get_biomass_tile_counts():
    '''
    The areas of the ecoregions are not perfectly reproducible because of the maxError in the intersection() function used to crop the ecoregions to the GEDI range
    The number of tiles for each ecoregion should be unaffected by this
    '''

    start_time = time.time()
    os.makedirs(f'{data_dir_path}/biomass', exist_ok=True)
    ecoregions_in_gedi_collection = get_ecoregions_in_gedi_collection() # collection of ecoregions cropped to the GEDI range
    biomes = sorted(set(ecoregions_in_gedi_collection.aggregate_array('BIOME_NAME').getInfo())) # list of 14 biomes
    biomes_ecoregions = {biome: {} for biome in biomes}
    num_tiles_per_biome = math.ceil(20000 / len(biomes)) # number of tiles per biome = 1,429
    intended_num_tiles = 0

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')

        ecoregions = sorted(ecoregions_in_gedi_collection.filter(ee.Filter.eq('BIOME_NAME', biome)).aggregate_array('ECO_NAME').getInfo()) # list of ecoregions in biome
        biomes_ecoregions[biome]['ecoregions'] = {}

        print('Calculating areas')

        for j, ecoregion in enumerate(ecoregions):
            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')

            biomes_ecoregions[biome]['ecoregions'][ecoregion] = {'area': ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion)).geometry().area().getInfo()} # saves the ecoregion's name and area

        biomes_ecoregions[biome]['area'] = sum([biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] for ecoregion in ecoregions]) # sets the biome area to the sum of its ecoregions' areas

        print('Calculating tile numbers')

        for j, ecoregion in enumerate(ecoregions):
            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')

            biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles'] = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion
            intended_num_tiles += biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles'] # adds the number of tiles for the ecoregion to the total number of tiles

    with open(f'{data_dir_path}/biomass/ecoregion_tile_counts.json', 'w') as file:
        json.dump(biomes_ecoregions, file, indent=4)

    print(f'Total number of tiles = {intended_num_tiles}') # 20,421 tiles
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}') # ~13 minutes

def generate_ecoregion_points(biome, ecoregion, num_ecoregion_tiles):
    start_time = time.time()
    print(f'{num_ecoregion_tiles} tile(s) in ecoregion')

    ecoregion_collection = get_ecoregions_in_gedi_collection().filter(ee.Filter.eq('ECO_NAME', ecoregion)) # extracts the collection for the ecoregion
    gedi_points_in_ecoregion = (utils.get_gedi_points(ecoregion_collection).map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                                                                           .randomColumn(seed=42) # adds a new property to each feature containing a random number
                                                                           .sort('random')) # sorts the features by the random numbers
    enough_points = False
    i = 0
    geojson_prefix = f'{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}_biomass_points'
    geojson_path = f'{data_dir_path}/biomass/points/{geojson_prefix}.geojson'

    while not enough_points:
        i += 1
        print(f'Sampling {i}x the number of needed points')
        task = ee.batch.Export.table.toCloudStorage(collection=gedi_points_in_ecoregion.limit(i*num_ecoregion_tiles), # selects points, possibly fewer than requested depending on how many points are available
                                                    bucket='biomass_ecoregion_data',
                                                    fileNamePrefix=geojson_prefix,
                                                    fileFormat='GeoJSON')
        task.start()
        print(f'Task ID: {task.id}')
        blob = bucket.blob(f'{geojson_prefix}.geojson')

        while not blob.exists():
            time.sleep(10) # waits 10 seconds

        blob.download_to_filename(geojson_path) # downloads the GeoJSON file from the bucket to the local path
        points = utils.read_geojson(geojson_path)['features']

        print(f'{len(points)} points before removing overlaps')
        available_number_of_points = len(points)

        if len(points) > 0:
            points_collection = ee.FeatureCollection(points).map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry())) # creates outer tiles around the points
            points = utils.remove_overlapping_tiles(points_collection.getInfo()['features']) # removes points with overlapping outer tiles
            print(f'{len(points)} points after removing overlaps')

            random.shuffle(points) # shuffles the points list
            points = points[:num_ecoregion_tiles] # selects the first num_ecoregion_tiles points

            if len(points) == num_ecoregion_tiles or available_number_of_points < num_ecoregion_tiles: # if there are enough points after removing overlaps or there are fewer points than desired in the ecoregion before removing overlaps
                enough_points = True
            else:
                blob.delete()
        else:
            break

    points = [{**{key: value for key, value in point.items() if key != 'id'}, 'properties': {key: value for key, value in point['properties'].items() if key != 'random'}} for point in points] # removes the 'id' and 'random' properties from the points and adds biome and ecoregion labels
    utils.save_geojson(points, path=geojson_path)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def generate_biomass_points():
    start_time = time.time()
    ecoregion_tile_counts = utils.read_json(f'{data_dir_path}/biomass/ecoregion_tile_counts.json')
    biomes = ecoregion_tile_counts.keys()

    for i, biome in enumerate(biomes):
        os.makedirs(f'{data_dir_path}/biomass/points/{biome.replace("/", "_")}', exist_ok=True)
        ecoregions = ecoregion_tile_counts[biome]['ecoregions'].keys()

        for j, ecoregion in enumerate(ecoregions):
            while utils.count_running_jobs() > 30: # if more than 30 jobs are running
                time.sleep(1) # checks again after 1 second

            print(f'Biome {i+1}/{len(biomes)}: {biome}')
            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')

            job_submitted = False

            while not job_submitted:
                result = subprocess.run(['sbatch',
                                        '-t', '3-00:00',
                                        '-p', partitions,
                                        '--mem', '500M',
                                        '--job-name', f'biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}',
                                        '-o', f'{data_dir_path}/biomass/output-files/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.out',
                                        '--account', 'davies_lab',
                                        'job.sh',
                                        env_path,
                                        'generate_biomass_points.py',
                                        'generate_ecoregion_points',
                                        biome.replace(' ', '_'),
                                        ecoregion.replace(' ', '_'),
                                        str(ecoregion_tile_counts[biome]['ecoregions'][ecoregion]['num_tiles'])])

                if result.returncode == 0:
                    job_submitted = True
                else:
                    time.sleep(50)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_biomass_points():
    ecoregion_tile_counts = utils.read_json(f'{data_dir_path}/biomass/ecoregion_tile_counts.json')
    biomes = ecoregion_tile_counts.keys()
    num_ecoregions = 0
    missing_ecoregions = []
    ecoregions_missing_tiles = {}

    for i, biome in enumerate(biomes):
        ecoregions = ecoregion_tile_counts[biome]['ecoregions'].keys()

        for j, ecoregion in enumerate(ecoregions):
            points_path = f'{data_dir_path}/biomass/points/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}_biomass_points.geojson'
            num_ecoregion_tiles = ecoregion_tile_counts[biome]['ecoregions'][ecoregion]['num_tiles']
            num_ecoregions += 1
            print_output = False

            if not os.path.exists(points_path):
                missing_ecoregions.append(ecoregion)
                print_output = True
                print(f'\nBiome {i+1}/{len(biomes)}: {biome}, Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} tiles do not exist ({num_ecoregion_tiles} tiles)')
            elif len(utils.read_geojson(points_path)['features']) < num_ecoregion_tiles:
                ecoregions_missing_tiles[ecoregion] = num_ecoregion_tiles - len(utils.read_geojson(points_path)['features'])
                print_output = True
                print(f'\nBiome {i+1}/{len(biomes)}: {biome}, Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} is missing {ecoregions_missing_tiles[ecoregion]} tile(s)')

            if print_output:
                with open(f'{data_dir_path}/biomass/output-files/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.out', 'r') as file:
                    print(file.read())

    print(f'\n{len(missing_ecoregions)}/{num_ecoregions} ecoregions are missing')
    print(f'{len(ecoregions_missing_tiles)}/{num_ecoregions} ecoregions are missing tiles')

def merge_ecoregion_points():
    points = []

    for biome in sorted(os.listdir(f'{data_dir_path}/biomass/points')):
        for ecoregion_points_file in sorted(os.listdir(f'{data_dir_path}/biomass/points/{biome}')):
            points += utils.read_geojson(f'{data_dir_path}/biomass/points/{biome}/{ecoregion_points_file}')['features']

    print(f'{len(points)} points before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    random.shuffle(points) # shuffles the points list
    points = [{**{key: value for key, value in point.items() if key != 'properties'}, 'id': i} for i, point in enumerate(points)] # assigns each point an ID
    print(f'{len(points)} points after removing overlaps')

    utils.save_geojson(features=points, path=f'{data_dir_path}/biomass/biomass_points.geojson') # saves the points

if __name__ == '__main__':
    if not os.path.exists(f'{data_dir_path}/biome_labels.json'):
        save_biome_ecoregion_labels() # assigns each biome and ecoregion an integer label

    if not os.path.exists(f'{data_dir_path}/biomass/ecoregion_tile_counts.json'):
        get_biomass_tile_counts() # calculates the number of tiles desired for each ecoregion

    if not bucket.exists():
        bucket = client.create_bucket(bucket) # creates a bucket for the ecoregion point data

    if len(argv) == 1:
        subprocess.run(['sbatch', '-t', '3-00:00', '-p', partitions, '--mem', '500M', '--job-name', 'generate_biomass_points', '-o', f'{data_dir_path}/biomass/output-files/generate_biomass_points.out', '--account', 'gajos_lab', 'job.sh', env_path, 'generate_biomass_points.py', 'generate_biomass_points'])
    if len(argv) > 1:
        if argv[1] == 'generate_biomass_points':
            generate_biomass_points()
        elif argv[1] == 'generate_ecoregion_points':
            generate_ecoregion_points(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '), num_ecoregion_tiles=int(argv[4]))
        elif argv[1] == 'check_biomass_points':
            check_biomass_points()
        elif argv[1] == 'merge_ecoregion_points':
            merge_ecoregion_points()
