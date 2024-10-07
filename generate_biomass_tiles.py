'''
generate_biomass_tiles.py by Lucia Gordon
A script to create a GeoJSON with biomass tiles balanced across biomes
'''

# imports
from sys import argv
import ee
import json
import math
import os
import shutil
import subprocess
import time
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment

def get_ecoregions_in_gedi_collection():
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

    return ecoregions_in_gedi_collection

def get_biomass_tile_counts():
    start_time = time.time()
    os.makedirs('biomes_ecoregions_data', exist_ok=True)
    ecoregions_in_gedi_collection = get_ecoregions_in_gedi_collection() # collection of ecoregions cropped to the GEDI range
    biomes = sorted([biome for biome in list(set(ecoregions_in_gedi_collection.aggregate_array('BIOME_NAME').getInfo())) if biome != 'N/A']) # list of biomes
    biomes_ecoregions = {biome: {} for biome in biomes}
    NUM_TILES = 20000 # total number of biomass tiles
    num_tiles_per_biome = math.ceil(NUM_TILES / len(biomes)) # total number of tiles per biome

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

            num_ecoregion_tiles = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion
            biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles'] = num_ecoregion_tiles

            print(f'{num_ecoregion_tiles} tile(s)')

    with open('biomes_ecoregions_data/biomes_ecoregions_biomass.json', 'w') as file:
        json.dump(biomes_ecoregions, file, indent=4)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_asset_existence(asset):
    try:
        var = asset.getInfo()

        return asset
    except ee.ee_exception.EEException as e:
        if 'not found' in str(e):
            print(e)

            return None
        else:
            return asset

def save_tiles(tiles, path):
    with open(path, 'w') as file:
        json.dump({'type': 'FeatureCollection', 'features': tiles}, file, indent=4) # save the tiles as a GeoJSON

    print(f'{len(tiles)} tile(s) made')

def generate_ecoregion_tiles(biome, ecoregion, num_ecoregion_tiles, tiles_from_points=True):
    start_time = time.time()

    print(f'{num_ecoregion_tiles} tile(s) in ecoregion')

    year = '2019'
    gedi_feature_collection = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                                 .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"')) # get feature collections with features in the selected year
    ecoregions_in_gedi_collection = get_ecoregions_in_gedi_collection()
    ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
    gedi_collection_names = (gedi_feature_collection
                             .filterBounds(ecoregion_collection) # get GEDI feature collections that have points within the ecoregion
                             .aggregate_array('table_id') # extract the IDs of the feature collections
                             .getInfo()) # list of names of the feature collections
    quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
    gedi_points = (ee.FeatureCollection([result for collection_name, result in ((collection_name, check_asset_existence(ee.FeatureCollection(collection_name))) for collection_name in gedi_collection_names) if result is not None]) # all GEDI feature collections with points in the ecoregion
    # gedi_points = (ee.FeatureCollection([ee.FeatureCollection(collection_name) for collection_name in gedi_collection_names]) # all GEDI feature collections with points in the ecoregion
                    .flatten() # merges all the feature collections into one
                    .filterBounds(ecoregion_collection) # collection of the GEDI points that are within the ecoregion
                    .filter(quality_filter) # filters for quality, growing season, and land
                    .map(lambda point: point.set('off_minus_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adds property for difference between leaf off and on days
                    .filter(ee.Filter.gt('off_minus_on', 0))) # only keeps points with leaf off after leaf on
    path = f'tiles/biomass/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'
    tiles = utils.read_geojson(path)['features'] if os.path.exists(path) else []
    num_tiles_to_make = num_ecoregion_tiles-len(tiles)
    TILE_SIZE = utils.read_yaml('config.yml')['TILE_SIZE']

    # gedi_points = (ee.FeatureCollection([result for collection_name, result in ((collection_name, check_asset_existence(ee.FeatureCollection(collection_name))) for collection_name in gedi_collection_names) if result is not None]) # all GEDI feature collections with points in the ecoregion
    #                 .flatten() # merges all the feature collections into one
    #                 .filterBounds(ecoregion_collection)
    #                 .filter(quality_filter)
    #                 ) # collection of the GEDI points that are within the ecoregion
    # print(gedi_points.size().getInfo())
    # exit()
    if tiles_from_points: # creates tiles centered on GEDI points
        tile_collection = (gedi_points.map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                                      .randomColumn('random') # adds a new property to each feature containing a random number
                                      .sort('random') # sorts the features by the random numbers
                                      .limit(num_tiles_to_make) # selects the first points
                                      .map(lambda point: point.buffer(TILE_SIZE / 2).bounds()) # converts points to tiles
                                      .getInfo()) 
        tiles += [{**{key: value for key, value in tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for tile in tile_collection['features']]
        save_tiles(tiles, path)
    else: # creates tiles randomly and checks if they contain GEDI points
        if os.path.exists(path):
            return

        seed = 0

        while len(tiles) < num_ecoregion_tiles:
            candidate_tiles = ee.FeatureCollection.randomPoints(region=ecoregion_collection, points=num_tiles_to_make, seed=seed).map(lambda point: point.buffer(TILE_SIZE / 2).bounds()).getInfo()['features'] # generates random tiles in the ecoregion
            tiles += [{**{key: value for key, value in candidate_tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for candidate_tile in candidate_tiles if gedi_points.filterBounds(candidate_tile['geometry']).size().getInfo() > 0] # save the tiles with at least one GEDI point
            seed += 1
            save_tiles(tiles, path)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def generate_biomass_tiles(tiles_from_points=True):
    start_time = time.time()
    biomes_ecoregions = utils.read_json('biomes_ecoregions_data/biomes_ecoregions_biomass.json')
    biomes = biomes_ecoregions.keys()
    hours = 2 if tiles_from_points else 6
    os.makedirs('tiles/biomass/ecoregion_tiles', exist_ok=True)

    print(f'{len(biomes)} biomes')

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')

        ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()
        no_ecoregions_missing = True

        for j, ecoregion in enumerate(ecoregions):
            path = f'tiles/biomass/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'
            num_ecoregion_tiles = biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles']

            if not os.path.exists(path) or len(utils.read_geojson(path)['features']) < num_ecoregion_tiles:
                print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')
                # generate_ecoregion_tiles(biome, ecoregion, num_ecoregion_tiles, tiles_from_points)
                subprocess.run(['sbatch', '-t', f'0-0{hours}:00:00', '-p', 'seas_compute,sapphire,tambe,shared', '-o', f'bash-outputs/{biome.replace("/", "_")}/{biome.replace("/", "_")}-{ecoregion.replace("/", "_")}.out', '-e', f'bash-errors/{biome.replace("/", "_")}/{biome.replace("/", "_")}-{ecoregion.replace("/", "_")}.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_ecoregion_tiles', biome.replace(' ', '_'), ecoregion.replace(' ', '_'), str(num_ecoregion_tiles), str(tiles_from_points)])
                no_ecoregions_missing = False

        if not no_ecoregions_missing:
            time.sleep(hours*60*60) # seconds

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_biomass_tiles():
    biomes_ecoregions = utils.read_json('biomes_ecoregions_data/biomes_ecoregions_biomass.json')
    biomes = biomes_ecoregions.keys()
    num_missing_ecoregions = 0
    num_ecoregions_missing_tiles = 0
    num_ecoregions = 0

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')

        ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()

        for j, ecoregion in enumerate(ecoregions):
            path = f'tiles/biomass/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'
            num_ecoregion_tiles = biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles']
            num_ecoregions += 1

            if not os.path.exists(path):
                print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} tiles do not exist')

                num_missing_ecoregions += 1
            elif len(utils.read_geojson(path)['features']) < num_ecoregion_tiles:
                print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} is missing tiles')

                num_ecoregions_missing_tiles += 1

    print(f'{num_missing_ecoregions}/{num_ecoregions} ecoregions are missing')
    print(f'{num_ecoregions_missing_tiles}/{num_ecoregions} ecoregions are missing tiles')

if __name__ == '__main__':
    if len(argv) == 1:
        subprocess.run(['sbatch', '-t', '4-00:00:00', '-p', 'seas_compute', '-o', f'bash-outputs/%A-%a.out', '-e', f'bash-errors/%A-%a.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_biomass_tiles'])
    else:
        if argv[1] == 'False': # tiles from points = False --> generate tiles randomly and check for points within tiles
            subprocess.run(['sbatch', '-t', '4-00:00:00', '-p', 'seas_compute', '-o', f'bash-outputs/%A-%a.out', '-e', f'bash-errors/%A-%a.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_biomass_tiles', 'False'])
        elif argv[1] == 'get_biomass_tile_counts':
            get_biomass_tile_counts()
        elif argv[1] == 'generate_biomass_tiles':
            if len(argv) == 2:
                generate_biomass_tiles()
            else:
                generate_biomass_tiles(tiles_from_points=utils.str_to_bool(argv[2]))
        elif argv[1] == 'generate_ecoregion_tiles':
            if len(argv) == 5:
                generate_ecoregion_tiles(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '), num_ecoregion_tiles=int(argv[4]))
            else:
                generate_ecoregion_tiles(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '), num_ecoregion_tiles=int(argv[4]), tiles_from_points=utils.str_to_bool(argv[5]))
        elif argv[1] == 'check_biomass_tiles':
            check_biomass_tiles()
