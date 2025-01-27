'''
generate_biomass_points.py by Lucia Gordon
A script to create a GeoJSON with biomass points balanced across biomes
'''

# imports
from matplotlib.lines import Line2D
# from shapely.geometry import mapping
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

            biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles'] = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion

    with open('biomes_ecoregions_data/biomes_ecoregions_biomass.json', 'w') as file:
        json.dump(biomes_ecoregions, file, indent=4)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

# def get_asset_if_valid(asset):
#     try:
#         asset.getInfo()

#         return asset
#     except ee.ee_exception.EEException as e:
#         if 'not found' in str(e):
#             print(e)

#             return None
#         else: # for any other kind of error
#             return asset

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
    gedi_points = (ee.FeatureCollection([result for _, result in ((collection_name, utils.get_asset_if_valid(ee.FeatureCollection(collection_name))) for collection_name in gedi_collection_names) if result is not None]) # all GEDI feature collections with points in the ecoregion
    # gedi_points = (ee.FeatureCollection([ee.FeatureCollection(collection_name) for collection_name in gedi_collection_names]) # all GEDI feature collections with points in the ecoregion
                    .flatten() # merges all the feature collections into one
                    .filterBounds(ecoregion_collection) # collection of the GEDI points that are within the ecoregion
                    .filter(quality_filter) # filters for quality, growing season, and land
                    .map(lambda point: point.set('off_minus_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adds property for difference between leaf off and on days
                    .filter(ee.Filter.gt('off_minus_on', 0))) # only keeps points with leaf off after leaf on

    return ecoregion_collection, gedi_points

def generate_ecoregion_points(biome, ecoregion, num_ecoregion_tiles, tiles_from_points=True):
    start_time = time.time()

    print(f'{num_ecoregion_tiles} tile(s) in ecoregion')
    print(f'Tiles from points = {tiles_from_points}')

    ecoregion_collection, gedi_points = get_gedi_points(ecoregion)
    points = []
    OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']

    if tiles_from_points: # creates tiles centered on GEDI points
        point_collection = (gedi_points.map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                                       .randomColumn('random') # adds a new property to each feature containing a random number
                                       .sort('random') # sorts the features by the random numbers
                                       .limit(num_ecoregion_tiles - len(points)) # selects the first points
                                       .map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry())) # creates outer tiles around the points
                                       .getInfo())
        points += [{**{key: value for key, value in point.items() if key != 'id'}, 'properties': {'outer_tile': point['properties']['outer_tile'], 'biome': biome, 'ecoregion': ecoregion}} for point in point_collection['features']]
    else: # creates tiles randomly and checks if they contain GEDI points
        seed = 0
        region = ecoregion_collection
        INNER_TILE_SIZE_M = utils.read_yaml('config.yml')['INNER_TILE_SIZE_M']

        try:
            ee.FeatureCollection.randomPoints(region=region, points=num_ecoregion_tiles, seed=seed).size().getInfo() # generates random points in the ecoregion
        except:
            region = ecoregion_collection.bounds()

            print("Using a simplified region due to the ecoregion geometry's complexity")

        while len(points) < num_ecoregion_tiles:
            candidate_points = (ee.FeatureCollection.randomPoints(region=region, points=num_ecoregion_tiles-len(points), seed=seed) # generates random points in the ecoregion
                                                    .map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry()))) # creates an outer tile around each point
            # print(f'{candidate_points.size().getInfo()} candidate point(s) generated')
            candidate_points = candidate_points.filter(ee.Filter.contains(leftValue=ecoregion_collection.geometry(), rightField='outer_tile')) # filters out points whose tiles are not contained in the ecoregion
            # print(f'{candidate_points.size().getInfo()} candidate point(s) after filtering for containment in ecoregion')
            seed += 1

            if candidate_points.size().getInfo() > 0:
                candidate_points = (candidate_points.map(lambda point: point.set('inner_tile', point.buffer(INNER_TILE_SIZE_M / 2).bounds().geometry())) # creates an inner tile around each point
                                                    .getInfo())
                candidate_points = [candidate_point for candidate_point in candidate_points['features'] if gedi_points.filterBounds(candidate_point['properties']['inner_tile']).size().getInfo() > 0] # saves the points with at least one GEDI point in their inner tile
                # print(f'{len(candidate_points)} candidate tile(s) after filtering for emptiness')
                points += [{**{key: value for key, value in candidate_point.items() if key != 'id'}, 'properties': {'outer_tile': candidate_point['properties']['outer_tile'], 'biome': biome, 'ecoregion': ecoregion}} for candidate_point in candidate_points] 
                print(f'{len(points)} points made so far')

    utils.save_geojson(points, path=f'biomass/points/ecoregion_points/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_points.geojson')

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def generate_biomass_points():
    start_time = time.time()
    biomes_ecoregions = utils.read_json('biomes_ecoregions_data/biomes_ecoregions_biomass.json')
    biomes = biomes_ecoregions.keys()
    complete = False
    os.makedirs('biomass/points/ecoregion_points', exist_ok=True)

    while not complete:
        complete = True

        for i, biome in enumerate(biomes):
            ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()

            for j, ecoregion in enumerate(ecoregions):
                points_path = f'biomass/points/ecoregion_points/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_points.geojson'
                run_ecoregion = True
                tiles_from_points = True

                if os.path.exists(points_path): # if there is a points file saved
                    run_ecoregion = False
                else: # if there is no points file saved
                    complete = False
                    error_path = f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err'

                    if os.path.exists(error_path): # if there is an error file
                        with open(error_path, 'r') as file: # opens error file
                            error = file.read()

                            if len(error) > 0: # if there is an error
                                print('\nError = {}'.format(error.split("\n")[-2]))

                                if 'Computation timed out' in error: # if the error is "computation timed out"
                                    tiles_from_points = False
                            else: # if there is no error
                                run_ecoregion = False

                if run_ecoregion:
                    while utils.count_running_jobs() > 40: # if more than 40 jobs are running
                        time.sleep(1) # checks again after 1 second

                    print(f'Biome {i+1}/{len(biomes)}: {biome}')
                    print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')
                    print(f'tiles_from_points = {tiles_from_points}')

                    hours = 3 if tiles_from_points else 120
                    num_ecoregion_tiles = biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles']
                    subprocess.run(['sbatch', '-t', f'0-0{hours}:00:00', '-p', partitions, '--mem', '500M', '--job-name', f'biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}', '-o', f'bash-outputs/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.out', '-e', f'bash-errors/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err', 'job.sh', env_path, 'generate_biomass_points.py', 'generate_ecoregion_points', biome.replace(' ', '_'), ecoregion.replace(' ', '_'), str(num_ecoregion_tiles), str(tiles_from_points)])
                    time.sleep(50) # checks again after 50 seconds since there is a delay between the job being submitted and the job running

        print(f'Complete = {complete}')

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def merge_ecoregion_points():
    os.makedirs('biomass/figures', exist_ok=True)

    ecoregion_point_filenames = os.listdir('biomass/points/ecoregion_points')
    points = []

    for filename in ecoregion_point_filenames:
        points_list = utils.read_geojson(f'biomass/points/ecoregion_points/{filename}')['features']
        points += points_list

    utils.save_geojson(features=points, path='biomass/points/biomass_points_overlapping.geojson')
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path='biomass/points/biomass_outer_tiles_overlapping.geojson')
    print(f'{len(points)} points before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    random.shuffle(points) # shuffles the points list
    points = [{**point, 'id': i} for i, point in enumerate(points)] # assigns each point an ID
    print(f'{len(points)} points after removing overlaps')

    utils.save_geojson(features=points, path='biomass/points/biomass_points.geojson') # saves the points
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path='biomass/points/biomass_outer_tiles.geojson')
    # utils.make_global_map(tiles=tiles, color='g', path='biomass/figures/biomass_map', title='Biomass') # plots the points on a global map

def check_biomass_points():
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
            path = f'biomass/points/ecoregion_points/biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}_biomass_points.geojson'
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
                    content = file.read()

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
        ecoregion_gdf.plot(ax=ax, edgecolor='blue', facecolor='blue', transform=ccrs.PlateCarree())

    for ecoregion in ecoregions_missing_tiles:
        ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
        ecoregion_gdf = gpd.GeoDataFrame.from_features(geemap.ee_to_geojson(ecoregion_collection))
        ecoregion_gdf.plot(ax=ax, edgecolor='red', facecolor='red', transform=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Missing ecoregions'),
                       Line2D([0], [0], color='red', lw=2, label='Ecoregions missing tiles')]

    ax.legend(handles=legend_elements, fontsize=6)
    plt.savefig('biomass/figures/ecoregions_missing_tiles.png', bbox_inches='tight')

if __name__ == '__main__':
    if not os.path.exists('biomes_ecoregions_data/biomes_ecoregions_biomass.json'):
        get_biomass_tile_counts() # takes ~13 minutes

    if len(argv) == 1:
        subprocess.run(['sbatch', '-t', '7-00:00:00', '-p', partitions, '--mem', '500M', '--job-name', 'generate_biomass_points', '-o', 'bash-outputs/generate_biomass_points.out', '-e', 'bash-errors/generate_biomass_points.err', 'job.sh', env_path, 'generate_biomass_points.py', 'generate_biomass_points'])
    if len(argv) > 1:
        if argv[1] == 'generate_biomass_points':
            generate_biomass_points()
        elif argv[1] == 'generate_ecoregion_points':
            generate_ecoregion_points(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '), num_ecoregion_tiles=int(argv[4]), tiles_from_points=utils.str_to_bool(argv[5]))
        elif argv[1] == 'merge_ecoregion_points':
            merge_ecoregion_points()
        elif argv[1] == 'check_biomass_points':
            check_biomass_points()
