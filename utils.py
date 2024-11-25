# imports
from shapely.geometry import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geojson
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import subprocess
import yaml

def count_running_jobs():    
    result = subprocess.run(['squeue', '--noheader', '--format=%u'], stdout=subprocess.PIPE, text=True) # gets running jobs
    user = subprocess.getoutput('whoami').strip()
    job_count = result.stdout.split().count(user) # counts the number of jobs for the user

    return job_count

def format_time(seconds):
    return f'{int(seconds // 60)} minute(s) {int(seconds % 60)} second(s)'

def get_last_day_of_month(month):
    return (datetime(int(year), month, 1) + relativedelta(months=1, days=-1)).day

def get_rectangle_center(coords):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return [center_x, center_y]

def make_global_map(tiles, color, path, title):
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(title, fontsize=8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)

    tile_centers = [get_rectangle_center(tile['geometry']['coordinates'][0]) for tile in tiles]

    for point in tile_centers:
        lon, lat = point
        plt.plot(lon, lat, marker='o', color=color, markeredgewidth=0, markersize=0.7, transform=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', bbox_inches='tight')

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
    tiles = [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': Polygon(tile['geometry']['coordinates'][0])} for tile in tiles]
    tiles_gdf = gpd.GeoDataFrame(tiles, geometry='geometry').reset_index(drop=False)
    intersections = gpd.sjoin(tiles_gdf, tiles_gdf)
    intersecting_indices = intersections[intersections['index_left'] != intersections['index_right']][['index_left', 'index_right']].to_numpy().tolist()
    indices_to_remove = []

    while len(intersecting_indices) > 0:
        index_to_remove = intersecting_indices[0][0]
        indices_to_remove.append(index_to_remove)
        intersecting_indices = [pair for pair in intersecting_indices if index_to_remove not in pair]

    indices_to_keep = [i for i in range(len(tiles)) if i not in indices_to_remove]
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
    '''convert a string input to a Boolean variable'''
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    return False
