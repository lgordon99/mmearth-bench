# imports
from shapely.geometry import mapping, Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
# observations = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_observations.tsv', sep='\t')
# sites = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_sites.tsv', sep='\t')
# profiles = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_profiles.tsv', sep='\t')
# layers = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_layers.tsv', sep='\t')
nitrogen = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_nitkjd.tsv', sep='\t')
organic_carbon = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_orgc.tsv', sep='\t')
pH = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_phaq.tsv', sep='\t')
upper_depth = 0
lower_depth = 5

nitrogen = nitrogen[(nitrogen['upper_depth'] >= upper_depth) & (nitrogen['lower_depth'] <= lower_depth)]
organic_carbon = organic_carbon[(organic_carbon['upper_depth'] >= upper_depth) & (organic_carbon['lower_depth'] <= lower_depth)]
pH = pH[(pH['upper_depth'] >= upper_depth) & (pH['lower_depth'] <= lower_depth)]
properties = {'nitrogen': {'dataframe': nitrogen, 'title': 'Nitrogen'}, 'organic_carbon': {'dataframe': organic_carbon, 'title': 'Organic carbon'}, 'pH': {'dataframe': pH, 'title': 'pH'}}
TILE_SIZE = utils.read_yaml('config.yml')['TILE_SIZE']

def generate_property_tiles(property_):
    os.makedirs(f'tiles/{property_}', exist_ok=True)
    dataframe = properties[property_]['dataframe']
    dataframe.to_csv(f'{data_dir_path}/{property_}.csv', index=False)
    tiles = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude]).buffer(TILE_SIZE / 2).bounds()).set({'value': measurement.value_avg}) for measurement in dataframe[i: i+5000].itertuples() if measurement.positional_uncertainty == 'Circa 100 m']).getInfo()['features'] for i in range(0, len(dataframe), 5000)])
    tiles = [{**{key: value for key, value in tile.items() if key != 'id'}} for tile in tiles]

    geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    with open(f'tiles/{property_}/{property_}_tiles_all.geojson', 'w') as file:
        json.dump(geojson_collection, file, indent=4) # save the tiles as a GeoJSON

    print(property_)
    print(f'{len(tiles)} tiles before removing overlaps')

    tiles = utils.remove_overlapping_tiles(tiles)
    print(f'{len(tiles)} tiles after removing overlaps')

    random.shuffle(tiles) # shuffling the tiles list
    tiles = [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': mapping(tile['geometry']), 'id': i} for i, tile in enumerate(tiles)]
    geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    with open(f'tiles/{property_}/{property_}_tiles.geojson', 'w') as file:
        json.dump(geojson_collection, file, indent=4) # save the tiles as a GeoJSON

    tile_centers = [utils.get_rectangle_center(tile['geometry']['coordinates'][0]) for tile in tiles]

    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(properties[property_]['title'], fontsize=8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)

    for point in tile_centers:
        lon, lat = point
        plt.plot(lon, lat, marker='o', color='red', markersize=0.3, transform=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    # plt.savefig(f'figures/map-{species}.pdf', bbox_inches='tight')
    plt.savefig(f'figures/map-{property_}.png', bbox_inches='tight')

for property_ in properties.keys():
    generate_property_tiles(property_)
