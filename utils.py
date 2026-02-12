# imports
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon
import ee
import geojson
import geopandas as gpd
import json
import numpy as np
import subprocess
import yaml

ee.Initialize(project='mmearth-bench') # initializes EE with our project

def count_running_jobs():
    result = subprocess.run(['squeue', '--noheader', '--format=%u'], stdout=subprocess.PIPE, text=True) # gets running jobs
    user = subprocess.getoutput('whoami').strip()
    job_count = result.stdout.split().count(user) # counts the number of jobs for the user

    return job_count

def count_running_tile_jobs():
    command = subprocess.run(['squeue', '-u', subprocess.getoutput('whoami').strip(), '-t', 'RUNNING', '--format=%.80j'], stdout=subprocess.PIPE, text=True)
    job_names = command.stdout.replace(' ', '').split('\n')[1:-1]
    count = len([job_name for job_name in job_names if 'get_tile' in job_name])

    return count

def format_time(seconds):
    days = int(seconds // 86400)  # number of seconds in a day
    seconds %= 86400
    hours = int(seconds // 3600)  # number of seconds in an hour
    seconds %= 3600
    minutes = int(seconds // 60)  # number of seconds in a minute
    seconds %= 60
    seconds = int(seconds)

    if days > 0:
        return f'{days} day(s), {hours} hour(s), {minutes} minute(s), {seconds} second(s)'
    if hours > 0:
        return f'{hours} hour(s), {minutes} minute(s), {seconds} second(s)'
    if minutes > 0:
        return f'{minutes} minute(s), {seconds} second(s)'
    else:
        return f'{seconds} second(s)'

def get_asset_if_valid(asset):
    try:
        asset.getInfo()

        return asset
    except ee.ee_exception.EEException as e:
        if 'not found' in str(e):
            print(e)

            return None
        else: # for any other kind of error
            return asset

def get_gedi_points(region):
    year = '2020'
    gedi_collection_names = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                               .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"') # feature collections with features in the selected year
                               .filterBounds(region) # GEDI feature collections that have points within the region
                               .aggregate_array('table_id') # extracts the IDs of the feature collections
                               .getInfo()) # list of names of the feature collections
    gedi_points = (ee.FeatureCollection([collection for collection in (get_asset_if_valid(ee.FeatureCollection(name)) for name in gedi_collection_names) if collection is not None]) # extracts the feature collections that are valid assets
                     .flatten() # merges all the feature collections into one
                     .filterBounds(region) # collection of the GEDI points that are within the region
                     .filter('degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0') # filters for quality, growing season, and land
                     .map(lambda point: point.set('leaf_off_day_minus_leaf_on_day', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adds property for difference between leaf off and on days
                     .filter(ee.Filter.gt('leaf_off_day_minus_leaf_on_day', 0)) # only keeps points whose leaf off day is after their leaf on day
                     .filter(ee.Filter.lte('agbd', 2000))) # filters by GEDI points with biomass value <= 2000

    return gedi_points

def get_rectangle_center(coords):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return [center_x, center_y]

def normalize(array):
    for i in range(array.shape[0]):
        valid_mask = array[i] != 65535

        if array[i][valid_mask].max() != array[i][valid_mask].min(): # check to avoid division by zero
            array[i][valid_mask] = (array[i][valid_mask] - array[i][valid_mask].min()) / (array[i][valid_mask].max() - array[i][valid_mask].min())
        else:
            array[i][valid_mask] = 0 # assigns a default value when all elements in the band are the same

    return array

def read_geojson(path):
    with open(path) as geojson_file:
        return geojson.load(geojson_file)

def read_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)

def read_yaml(path):
    with open(path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def remove_overlapping_tiles(tiles):
    tiles_for_gdf = [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': Polygon(tile['properties']['outer_tile']['coordinates'][0])} for tile in tiles] # replaces the geometry from the point to the outer tile
    tiles_gdf = gpd.GeoDataFrame(tiles_for_gdf, geometry='geometry').reset_index(drop=False) # creates a GeoDataFrame with an index column for tracking
    intersections = gpd.sjoin(tiles_gdf, tiles_gdf) # all pairs of tiles that intersect
    intersecting_indices = np.array(intersections[intersections['index_left'] != intersections['index_right']][['index_left', 'index_right']].to_numpy().tolist()) # filters out self-intersections
    indices_to_remove = []

    while len(intersecting_indices) > 0: # while there are intersecting tiles
        conflicting_indices, num_conflicts = np.unique(intersecting_indices, return_counts=True)
        index_to_remove = conflicting_indices[np.argmax(num_conflicts)]
        indices_to_remove.append(index_to_remove)
        intersecting_indices = intersecting_indices[~((intersecting_indices == index_to_remove).any(axis=1))]

    indices_to_keep = [i for i in range(len(tiles)) if i not in indices_to_remove] # indices of non-overlapping tiles
    tiles = [tiles[i] for i in indices_to_keep]

    return tiles

def save_geojson(features, path):
    with open(path, 'w') as file:
        json.dump({'type': 'FeatureCollection', 'features': features}, file, indent=4)

    if len(features) == 1:
        print(f'{len(features)} item saved')
    else:
        print(f'{len(features)} items saved')

def str_to_bool(string):
    '''Convert a string input to a Boolean variable'''

    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    return False
