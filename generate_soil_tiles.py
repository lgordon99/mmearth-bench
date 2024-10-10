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
nitrogen = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_nitkjd.tsv', sep='\t')
organic_carbon = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_orgc.tsv', sep='\t', low_memory=False)
pH = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_phaq.tsv', sep='\t')
upper_depth = 0
lower_depth = 5
nitrogen = nitrogen[(nitrogen['upper_depth'] >= upper_depth) & (nitrogen['lower_depth'] <= lower_depth)]
organic_carbon = organic_carbon[(organic_carbon['upper_depth'] >= upper_depth) & (organic_carbon['lower_depth'] <= lower_depth)]
pH = pH[(pH['upper_depth'] >= upper_depth) & (pH['lower_depth'] <= lower_depth)]
properties = {'soil_nitrogen': {'dataframe': nitrogen, 'title': 'Soil nitrogen', 'color': 'b'},
              'soil_organic_carbon': {'dataframe': organic_carbon, 'title': 'Soil organic carbon', 'color': 'tab:brown'},
              'soil_pH': {'dataframe': pH, 'title': 'Soil pH', 'color': 'tab:purple'}}
TILE_SIZE = utils.read_yaml('config.yml')['TILE_SIZE']

def generate_property_tiles(property_):
    os.makedirs(f'{property_}/figures', exist_ok=True)

    dataframe = properties[property_]['dataframe']
    dataframe.to_csv(f'{data_dir_path}/{property_}.csv', index=False)
    tiles = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude]).buffer(TILE_SIZE / 2).bounds()).set({'value': measurement.value_avg}) for measurement in dataframe[i: i+5000].itertuples() if measurement.positional_uncertainty == 'Circa 100 m']).getInfo()['features'] for i in range(0, len(dataframe), 5000)])
    tiles = [{**{key: value for key, value in tile.items() if key != 'id'}} for tile in tiles]

    geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    with open(f'{property_}/{property_}_tiles_overlapping.geojson', 'w') as file:
        json.dump(geojson_collection, file, indent=4) # save the tiles as a GeoJSON

    print(properties[property_]['title'])
    print(f'{len(tiles)} tiles before removing overlaps')

    tiles = utils.remove_overlapping_tiles(tiles)
    print(f'{len(tiles)} tiles after removing overlaps')

    random.shuffle(tiles) # shuffling the tiles list
    tiles = [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': mapping(tile['geometry']), 'id': i} for i, tile in enumerate(tiles)]
    utils.save_geojson(features=tiles, path=f'{property_}/{property_}_tiles.geojson')
    # geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    # with open(f'{property_}/{property_}_tiles.geojson', 'w') as file:
    #     json.dump(geojson_collection, file, indent=4) # save the tiles as a GeoJSON

    utils.make_global_map(tiles=tiles, color=properties[property_]['color'], path=f'{property_}/figures/{property_}_map', title=properties[property_]['title'])

    # fig = plt.figure(dpi=300)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_title(properties[property_]['title'], fontsize=8)
    # ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
    # ax.add_feature(cfeature.COASTLINE)

    # tile_centers = [utils.get_rectangle_center(tile['geometry']['coordinates'][0]) for tile in tiles]

    # for point in tile_centers:
    #     lon, lat = point
    #     plt.plot(lon, lat, marker='o', color=properties[property_]['color'], markeredgewidth=0, markersize=0.7, transform=ccrs.PlateCarree())

    # ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    # plt.savefig(f'{property_}/figures/{property_}_map.pdf', bbox_inches='tight')
    # plt.savefig(f'{property_}/figures/{property_}_map.png', bbox_inches='tight')

for property_ in properties.keys():
    generate_property_tiles(property_)
