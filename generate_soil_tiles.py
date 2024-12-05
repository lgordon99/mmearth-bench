# imports
from shapely.geometry import mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

# read in data
nitrogen = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_nitkjd.tsv', sep='\t')
organic_carbon = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_orgc.tsv', sep='\t', low_memory=False)
pH = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_phaq.tsv', sep='\t')

# depth range
upper_depth = 0 # cm
lower_depth = 5 # cm

# get measurements in selected depth range
nitrogen = nitrogen[(nitrogen['upper_depth'] >= upper_depth) & (nitrogen['lower_depth'] <= lower_depth)]
organic_carbon = organic_carbon[(organic_carbon['upper_depth'] >= upper_depth) & (organic_carbon['lower_depth'] <= lower_depth)]
pH = pH[(pH['upper_depth'] >= upper_depth) & (pH['lower_depth'] <= lower_depth)]

properties = {'soil_nitrogen': {'dataframe': nitrogen, 'title': 'Soil nitrogen', 'color': 'b'},
              'soil_organic_carbon': {'dataframe': organic_carbon, 'title': 'Soil organic carbon', 'color': 'tab:brown'},
              'soil_pH': {'dataframe': pH, 'title': 'Soil pH', 'color': 'tab:purple'}}
TILE_SIZE = utils.read_yaml('config.yml')['TILE_SIZE']

def generate_property_tiles(property_):
    print(properties[property_]['title'])

    dataframe = properties[property_]['dataframe']
    dataframe.to_csv(f'{data_dir_path}/{property_}.csv', index=False) # saves the dataframe as a CSV
    tiles = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude]).buffer(TILE_SIZE / 2).bounds()).set({'value': measurement.value_avg}) for measurement in dataframe[i: i+5000].itertuples() if measurement.positional_uncertainty == 'Circa 100 m']).getInfo()['features'] for i in range(0, len(dataframe), 5000)])
    tiles = [{**{key: value for key, value in tile.items() if key != 'id'}} for tile in tiles] # removes ID property

    print(f'{len(tiles)} tiles before removing overlaps')

    os.makedirs(f'{property_}/tiles', exist_ok=True)
    utils.save_geojson(features=tiles, path=f'{property_}/tiles/{property_}_tiles_overlapping.geojson') # saves all the tiles as a GeoJSON

    tiles = utils.remove_overlapping_tiles(tiles) # removes overlapping tiles
    print(f'{len(tiles)} tiles after removing overlaps')

    random.shuffle(tiles) # shuffles the tiles list
    tiles = [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': mapping(tile['geometry']), 'id': i} for i, tile in enumerate(tiles)] # gives each tile an ID and converts the geometry to dictionary format
    utils.save_geojson(features=tiles, path=f'{property_}/tiles/{property_}_tiles.geojson') # saves the tiles as a GeoJSON

    os.makedirs(f'{property_}/figures', exist_ok=True)
    utils.make_global_map(tiles=tiles, color=properties[property_]['color'], path=f'{property_}/figures/{property_}_map', title=properties[property_]['title'])

for property_ in properties.keys():
    generate_property_tiles(property_)
