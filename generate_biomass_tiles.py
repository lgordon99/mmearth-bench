'''
generate_biomass_tiles.py by Lucia Gordon
A script to create a GeoJSON with biomass tiles balanced across biomes
'''

# imports
from matplotlib.lines import Line2D
from shapely.geometry import mapping
from sys import argv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import geemap
import geopandas as gpd
import json
import math
import matplotlib.pyplot as plt
import os
import random
import shutil
import subprocess
import time
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)

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
        asset_data = asset.getInfo()

        return asset
    except ee.ee_exception.EEException as e:
        if 'not found' in str(e):
            print(e)

            return None
        else: # for any other kind of error
            return asset

def get_gedi_points(ecoregion):
    year = '2020'
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

    return ecoregion_collection, gedi_points

def generate_ecoregion_tiles(biome, ecoregion, num_ecoregion_tiles, tiles_from_points=True):
    start_time = time.time()

    print(f'{num_ecoregion_tiles} tile(s) in ecoregion')

    ecoregion_collection, gedi_points = get_gedi_points(ecoregion)
    path = f'biomass/tiles/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'
    tiles = utils.read_geojson(path)['features'] if os.path.exists(path) else []
    TILE_SIZE = utils.read_yaml('config.yml')['TILE_SIZE']

    if tiles_from_points: # creates tiles centered on GEDI points
        tile_collection = (gedi_points.map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                                      .randomColumn('random') # adds a new property to each feature containing a random number
                                      .sort('random') # sorts the features by the random numbers
                                      .limit(num_ecoregion_tiles - len(tiles)) # selects the first points
                                      .map(lambda point: point.buffer(TILE_SIZE / 2).bounds()) # converts points to tiles
                                      .getInfo()) 
        tiles += [{**{key: value for key, value in tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for tile in tile_collection['features']]
        utils.save_geojson(tiles, path)
    else: # creates tiles randomly and checks if they contain GEDI points
        if os.path.exists(path):
            return

        seed = 0

        while len(tiles) < num_ecoregion_tiles:
            candidate_tiles = ee.FeatureCollection.randomPoints(region=ecoregion_collection, points=num_ecoregion_tiles - len(tiles), seed=seed).map(lambda point: point.buffer(TILE_SIZE / 2).bounds()).getInfo()['features'] # generates random tiles in the ecoregion
            tiles += [{**{key: value for key, value in candidate_tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for candidate_tile in candidate_tiles if gedi_points.filterBounds(candidate_tile['geometry']).size().getInfo() > 0] # save the tiles with at least one GEDI point
            seed += 1

        utils.save_geojson(tiles, path)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def generate_biomass_tiles():
    start_time = time.time()
    biomes_ecoregions = utils.read_json('biomes_ecoregions_data/biomes_ecoregions_biomass.json')
    biomes = biomes_ecoregions.keys()
    run = True
    no_error_files = True
    error_found = False
    tiles_from_points = True
    os.makedirs('biomass/tiles/ecoregion_tiles', exist_ok=True)

    print(f'{len(biomes)} biomes')

    while run:
        # while utils.count_running_jobs() > 1: # if jobs other than this are running
        #     time.sleep(1) # checks again after 1 second

        for biome in biomes:
            ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()

            for ecoregion in ecoregions:
                error_path = f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err'

                if os.path.exists(error_path): # if there is an error file
                    no_error_files = False

                    with open(error_path, 'r') as file: # opens error file
                        if len(file.read().replace('*** Earth Engine *** Share your feedback by taking our Annual Developer Satisfaction Survey: https://google.qualtrics.com/jfe/form/SV_0JLhFqfSY1uiEaW?source=Init\n', '')) > 0: # if there is an error
                            error_found = True
                            break

            if error_found:
                break

        if no_error_files or error_found:
            print(f'no_error_files={no_error_files}, error_found={error_found}')

            for i, biome in enumerate(biomes):
                ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()

                for j, ecoregion in enumerate(ecoregions):
                    tiles_path = f'biomass/tiles/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'
                    error_path = f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err'
                    num_ecoregion_tiles = biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles']
                    run_ecoregion = False

                    if not os.path.exists(tiles_path): # if there is no tiles file saved
                        if not os.path.exists(error_path): # if there is no error file
                            tiles_from_points = True
                            run_ecoregion = True
                        else: # if there is an error file
                            with open(error_path, 'r') as file: # opens error file
                                content = file.read().replace('*** Earth Engine *** Share your feedback by taking our Annual Developer Satisfaction Survey: https://google.qualtrics.com/jfe/form/SV_0JLhFqfSY1uiEaW?source=Init\n', '')

                                if len(content) > 0: # if there is an error
                                    run_ecoregion = True
                                    print(f'Error = {content}')

                                    if 'Computation timed out' not in content: # if the error is not "computation timed out"
                                        tiles_from_points = True
                                    else: # if the error is "computation timed out"
                                        tiles_from_points = False
                                else: # if there is no error
                                    run_ecoregion = False

                        if run_ecoregion:
                            while utils.count_running_jobs() > 40: # if more than 40 jobs are running
                                time.sleep(1) # checks again after 1 second

                            print(f'Biome {i+1}/{len(biomes)}: {biome}')
                            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')
                            print(f'tiles_from_points={tiles_from_points}\n')
                            hours = 3 if tiles_from_points else 70
                            subprocess.run(['sbatch', '-t', f'0-0{hours}:00:00', '-p', partitions, '--job-name', f'biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}', '-o', f'bash-outputs/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.out', '-e', f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_ecoregion_tiles', biome.replace(' ', '_'), ecoregion.replace(' ', '_'), str(num_ecoregion_tiles), str(tiles_from_points)])
                            time.sleep(50) # checks again after 50 seconds since there is a delay between the job being submitted and the job running
        # else:
        #     run = False

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def merge_ecoregion_tiles():
    os.makedirs('biomass/figures', exist_ok=True)

    ecoregion_tile_filenames = os.listdir('biomass/tiles/ecoregion_tiles')
    tiles = []

    for filename in ecoregion_tile_filenames:
        tiles_list = utils.read_geojson(f'biomass/tiles/ecoregion_tiles/{filename}')['features']
        tiles += tiles_list

    utils.save_geojson(features=tiles, path='biomass/tiles/biomass_tiles_overlapping.geojson')

    tiles = utils.remove_overlapping_tiles(tiles) # removes overlapping tiles

    random.shuffle(tiles) # shuffles the tiles list
    tiles = [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': mapping(tile['geometry']), 'id': i} for i, tile in enumerate(tiles)] # assigns each tile an ID

    utils.save_geojson(features=tiles, path='biomass/tiles/biomass_tiles.geojson') # saves the tiles
    utils.make_global_map(tiles=tiles, color='g', path='biomass/figures/biomass_map', title='Biomass') # plots the points on a global map

def check_biomass_tiles():
    biomes_ecoregions = utils.read_json('biomes_ecoregions_data/biomes_ecoregions_biomass.json')
    biomes = biomes_ecoregions.keys()
    num_missing_ecoregions = 0
    num_ecoregions_missing_tiles = 0
    num_ecoregions = 0
    missing_ecoregions = []
    ecoregions_missing_tiles = []

    for i, biome in enumerate(biomes):
        ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()
        print(f'\nBiome {i+1}/{len(biomes)}: {biome}')

        for j, ecoregion in enumerate(ecoregions):
            path = f'biomass/tiles/ecoregion_tiles/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson'
            num_ecoregion_tiles = biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles']
            num_ecoregions += 1

            if not os.path.exists(path):
                num_missing_ecoregions += 1
                missing_ecoregions.append(ecoregion)

                print(f'\nEcoregion {j+1}/{len(ecoregions)}: {ecoregion} tiles do not exist ({num_ecoregion_tiles} tiles)')

                with open(f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err', 'r') as file:
                    content = file.read().replace('*** Earth Engine *** Share your feedback by taking our Annual Developer Satisfaction Survey: https://google.qualtrics.com/jfe/form/SV_0JLhFqfSY1uiEaW?source=Init\n', '')

                    if len(content) > 0:
                        print(content.split('\n')[-2])

            elif len(utils.read_geojson(path)['features']) < num_ecoregion_tiles:
                num_ecoregions_missing_tiles += 1

                if len(utils.read_geojson(path)['features']) == 0:
                    missing_ecoregions.append(ecoregion)
                else:
                    ecoregions_missing_tiles.append(ecoregion)

                # _, gedi_points = get_gedi_points(ecoregion)

                with open(f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err', 'r') as file:
                    content = file.read().replace('*** Earth Engine *** Share your feedback by taking our Annual Developer Satisfaction Survey: https://google.qualtrics.com/jfe/form/SV_0JLhFqfSY1uiEaW?source=Init\n', '')

                    # if len(content) > 0:
                    #     print(content.split('\n')[-2])

                # print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} is missing tiles ({len(utils.read_geojson(path)['features'])}/{num_ecoregion_tiles} tile(s) made, {gedi_points.size().getInfo()} GEDI points)')
                # print(f'\nEcoregion {j+1}/{len(ecoregions)}: {ecoregion} is missing tiles ({len(utils.read_geojson(path)['features'])}')

    print(f'\n{num_missing_ecoregions}/{num_ecoregions} ecoregions are missing')
    print(f'{num_ecoregions_missing_tiles}/{num_ecoregions} ecoregions are missing tiles')

    ecoregions_in_gedi_collection = get_ecoregions_in_gedi_collection()
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title('Ecoregions with Missing Tiles', fontsize=8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=0.3)

    for ecoregion in missing_ecoregions:
        ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
        ecoregion_gdf = gpd.GeoDataFrame.from_features(geemap.ee_to_geojson(ecoregion_collection))
        ecoregion_gdf.plot(ax=ax, edgecolor='blue', facecolor='blue', transform=ccrs.PlateCarree())  # Outline in black, transparent fill

    for ecoregion in ecoregions_missing_tiles:
        ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
        ecoregion_gdf = gpd.GeoDataFrame.from_features(geemap.ee_to_geojson(ecoregion_collection))
        ecoregion_gdf.plot(ax=ax, edgecolor='red', facecolor='red', transform=ccrs.PlateCarree())  # Outline in black, transparent fill

    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Missing ecoregions'),
                       Line2D([0], [0], color='red', lw=2, label='Ecoregions missing tiles')]

    ax.legend(handles=legend_elements, fontsize=6)
    plt.savefig('figures/ecoregions_missing_tiles.png', bbox_inches='tight')

if __name__ == '__main__':
    if len(argv) == 1:
        subprocess.run(['sbatch', '-t', '3-00:00:00', '-p', partitions, '--job-name', 'generate_biomass_tiles', '-o', 'bash-outputs/generate_biomass_tiles.out', '-e', 'bash-errors/generate_biomass_tiles.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_biomass_tiles'])
    else:
        if argv[1] == 'False': # tiles from points = False --> generate tiles randomly and check for points within tiles
            subprocess.run(['sbatch', '-t', '3-00:00:00', '-p', partitions, '-o', 'bash-outputs/generate_biomass_tiles_randomly.out', '-e', 'bash-errors/generate_biomass_tiles_randomly.err', 'job.sh', env_path, 'generate_biomass_tiles.py', 'generate_biomass_tiles', 'False'])
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
        elif argv[1] == 'merge_ecoregion_tiles':
            merge_ecoregion_tiles()
        elif argv[1] == 'check_biomass_tiles':
            check_biomass_tiles()
