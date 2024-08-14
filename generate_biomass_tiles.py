'''
generate_biomass_tiles.py by Lucia Gordon
A script to create a GeoJSON with tiles sampled from various ecoregions and biomes of the world
'''

# imports
from sys import argv
from utils import format_time, read_json, read_yaml, str_to_bool
import ee
import json
import math
import os
import shutil
import subprocess
import time

ee.Initialize(project='mmearth-bench') # initializes EE with our project
env_path = read_yaml('config-user.yml')['env_path']

def get_biomes_ecoregions_area(ecoregions_collection, title):
    os.makedirs('biomes_ecoregions_data', exist_ok=True)
    ecoregions = ecoregions_collection.aggregate_array('ECO_NAME').getInfo()
    biomes_ecoregions = {}

    for i, ecoregion in enumerate(ecoregions):
        print(f'Ecoregion {i+1} of {len(ecoregions)}')

        ecoregion_feature = ecoregions_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
        biome = ecoregion_feature.getInfo()['features'][0]['properties']['BIOME_NAME']

        if biome != 'N/A':
            if biome not in biomes_ecoregions.keys():
                biomes_ecoregions[biome] = {}
                biomes_ecoregions[biome]['ecoregions'] = {}

            biomes_ecoregions[biome]['ecoregions'][ecoregion] = {'area': ecoregion_feature.geometry().area().getInfo()}

            for biome in biomes_ecoregions.keys():
                biomes_ecoregions[biome]['area'] = sum([biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] for ecoregion in biomes_ecoregions[biome]['ecoregions'].keys()])

            with open(f'biomes_ecoregions_data/{title}', 'w') as f:
                json.dump(biomes_ecoregions, f, indent=4)

def get_areas_biomass():
    start_time = time.time()
    ecoregions_collection = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
    gedi_latitude_range = [-51.6, 51.6] # GEDI covers the latitude band between 51.6 degees N and S
    gedi_range_polygon = ee.Geometry.Polygon(coords=[[-180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[1]],
                                                    [-180, gedi_latitude_range[1]]],
                                            proj=None,
                                            geodesic=False) # polygon covering GEDI range
    ecoregions_in_gedi_collection = (ecoregions_collection
                                     .filterBounds(gedi_range_polygon) # exclude ecoregions outside of the GEDI range
                                     .map(lambda feature: feature.setGeometry(feature.geometry().intersection(gedi_range_polygon, maxError=1)))) # crop ecoregions to the GEDI range
    get_biomes_ecoregions_area(ecoregions_collection=ecoregions_in_gedi_collection, title='biomes_ecoregions_gedi.json')

    print(format_time(seconds=time.time()-start_time))

def generate_ecoregion_tiles(biome, ecoregion, tiles_from_points=True):
    start_time = time.time()
    ecoregions_collection = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
    gedi_latitude_range = [-51.6, 51.6] # GEDI covers the latitude band between 51.6 degees N and S
    gedi_range_polygon = ee.Geometry.Polygon(coords=[[-180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[1]],
                                                    [-180, gedi_latitude_range[1]]],
                                             proj=None,
                                             geodesic=False) # polygon covering GEDI range
    ecoregions_in_gedi_collection = (ecoregions_collection
                                     .filterBounds(gedi_range_polygon) # exclude ecoregions outside of the GEDI range
                                     .map(lambda feature: feature.setGeometry(feature.geometry().intersection(gedi_range_polygon, maxError=1)))) # crop ecoregions to the GEDI range
    year = '2020'
    gedi_feature_collection = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                                 .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"')) # get feature collections with features in the selected year
    ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
    gedi_collection_names = (gedi_feature_collection
                             .filterBounds(ecoregion_collection) # get GEDI feature collections that have points within the ecoregion
                             .aggregate_array('table_id') # extract the IDs of the feature collections
                             .getInfo()) # list of names of the feature collections
    quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
    biomes_ecoregions = read_json('biomes_ecoregions_data/biomes_ecoregions_gedi.json')
    biomes = biomes_ecoregions.keys()
    NUM_TILES = 20000
    num_tiles_per_biome = math.ceil(NUM_TILES / len(biomes))
    num_ecoregion_tiles = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion
    print(f'{num_ecoregion_tiles} tiles in ecoregion')
    TILE_SIZE = read_yaml('config.yml')['TILE_SIZE']

    if tiles_from_points:
        tiles = (ee.FeatureCollection([ee.FeatureCollection(name) for name in gedi_collection_names])
                    .flatten() # merge all the feature collections into one
                    .filterBounds(ecoregion_collection) # collection of the GEDI points that are within the ecoregion
                    .filter(quality_filter) # filters for quality, growing season, and land
                    .map(lambda point: point.set('off_after_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
                    .filter(ee.Filter.gt('off_after_on', 0)) # only keep points with leaf off after leaf on
                    .map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                    .randomColumn('random') # add a new property to each feature containing a random number
                    .sort('random') # sort the features by the random numbers
                    .limit(num_ecoregion_tiles) # select the first points
                    .map(lambda point: point.buffer(TILE_SIZE / 2).bounds())) # convert points to tiles
        tiles = [{**{key: value for key, value in tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for tile in tiles.getInfo()['features']]
    else:
        gedi_points = (ee.FeatureCollection([ee.FeatureCollection(collection_name) for collection_name in gedi_collection_names]) # all GEDI feature collections with points in the ecoregion
                         .flatten() # merge all the feature collections into one
                         .filterBounds(ecoregion_collection) # collection of the GEDI points that are within the ecoregion
                         .filter(quality_filter) # filters for quality, growing season, and land
                         .map(lambda point: point.set('off_minus_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
                         .filter(ee.Filter.gt('off_minus_on', 0))) # only keep points with leaf off after leaf on
        tiles = []

        while len(tiles) < num_ecoregion_tiles:
            candidate_tiles = ee.FeatureCollection.randomPoints(region=ecoregion_collection, points=num_ecoregion_tiles).map(lambda point: point.buffer(TILE_SIZE / 2).bounds()).getInfo()['features'] # generate random tiles in the ecoregion
            tiles += [{**{key: value for key, value in candidate_tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for candidate_tile in candidate_tiles if gedi_points.filterBounds(candidate_tile['geometry']).size().getInfo() > 0] # save the tiles with at least one GEDI point
            print(f'{len(tiles)} tile(s) made so far')

    print(f'{len(tiles)} tile(s) made')

    geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    with open(f'tiles/biomass/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson', 'w') as f:
        json.dump(geojson_collection, f, indent=4) # save the tiles as a GeoJSON

    print(f'Time taken: {format_time(seconds=time.time()-start_time)}')

def generate_biomass_tiles(tiles_from_points=True):
    start_time = time.time()
    biomes_ecoregions = read_json('biomes_ecoregions_data/biomes_ecoregions_gedi.json')
    biomes = biomes_ecoregions.keys()
    os.makedirs('tiles/biomass/ecoregion_tiles', exist_ok=True)

    print(f'{len(biomes)} biomes')

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')
        ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()

        for j, ecoregion in enumerate(ecoregions):
            none_missing = True

            if not os.path.exists(f'tiles/biomass/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'):
                print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')
                # generate_ecoregion_tiles(biome, ecoregion, tiles_from_points)
                subprocess.run(['sbatch', '-t', '0-00:30:00', '-p', 'seas_compute,sapphire,tambe,shared', '-o', f'bash-outputs/{biome.replace("/", "_")}/{biome.replace("/", "_")}-{ecoregion.replace("/", "_")}.out', '-e', f'bash-errors/{biome.replace("/", "_")}/{biome.replace("/", "_")}-{ecoregion.replace("/", "_")}.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_ecoregion_tiles', biome.replace(' ', '_'), ecoregion.replace(' ', '_'), str(tiles_from_points)])
                none_missing = False

        if not none_missing:
            time.sleep(30*60) # seconds

    print(f'Time taken: {format_time(seconds=time.time()-start_time)}')

def check_biomass_tiles():
    biomes_ecoregions = read_json('biomes_ecoregions_data/biomes_ecoregions_gedi.json')
    biomes = biomes_ecoregions.keys()
    count_not_done = 0
    num_ecoregions = 0

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')

        ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()

        for j, ecoregion in enumerate(ecoregions):
            num_ecoregions += 1

            if not os.path.exists(f'tiles/biomass/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'):
                print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} tiles do not exist')
                count_not_done += 1

    print(f'Tiles have not been made for {count_not_done}/{num_ecoregions} ecoregions')

if __name__ == '__main__':
    if not os.path.exists('biomes_ecoregions_data/biomes_ecoregions_gedi.json'):
        subprocess.run(['sbatch', 'job.sh', env_path, 'generate_biomass_tiles.py', 'get_areas_biomass'])

    if len(argv) == 1:
        subprocess.run(['sbatch', '-t', '1-00:00:00', '-p', 'seas_compute', '-o', f'bash-outputs/%A-%a.out', '-e', f'bash-errors/%A-%a.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_biomass_tiles'])
    else:
        if argv[1] == 'False': # tiles from points = False --> generate tiles randomly and check for points within
            subprocess.run(['sbatch', '-t', '1-00:00:00', '-p', 'seas_compute', '-o', f'bash-outputs/%A-%a.out', '-e', f'bash-errors/%A-%a.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_biomass_tiles', 'False'])
        elif argv[1] == 'get_areas_biomass':
            get_areas_biomass()
        elif argv[1] == 'generate_biomass_tiles':
            if len(argv) == 2:
                generate_biomass_tiles()
            else:
                generate_biomass_tiles(tiles_from_points=str_to_bool(argv[2]))
        elif argv[1] == 'generate_ecoregion_tiles':
            if len(argv) == 4:
                generate_ecoregion_tiles(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '))
            else:
                generate_ecoregion_tiles(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '), tiles_from_points=str_to_bool(argv[4]))
        elif argv[1] == 'check_biomass_tiles':
            check_biomass_tiles()
