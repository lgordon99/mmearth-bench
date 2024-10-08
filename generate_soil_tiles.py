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
observations = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_observations.tsv', sep='\t')
sites = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_sites.tsv', sep='\t')
profiles = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_profiles.tsv', sep='\t')
layers = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_layers.tsv', sep='\t')
nitrogen = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_nitkjd.tsv', sep='\t')
organic_carbon = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_orgc.tsv', sep='\t')
pH = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_phaq.tsv', sep='\t')
water_retention = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_wg1500.tsv', sep='\t')

# silt = pd.read_csv(f'{data_dir_path}/WoSIS_2023_December/wosis_202312_silt.tsv', sep='\t')
# vals = silt['value_avg'].values
# print(min(vals), max(vals))

def get_site_id(profile_id):
    return layers[layers['profile_id'] == profile_id]['site_id'].unique()[0]

# print(observations.head(), sites.head(), profiles.head(), layers.head(), nitrogen.head())
# print(f'Columns = {observations.columns.tolist()}')
# # print(f'Columns = {sites.columns.tolist()}')
# # print(f'Columns = {profiles.columns.tolist()}')
# print(f'Columns = {layers.columns.tolist()}')
print(f'Columns = {nitrogen.columns.tolist()}')

upper_depth = 0
lower_depth = 5

nitrogen = nitrogen[(nitrogen['upper_depth'] >= upper_depth) & (nitrogen['lower_depth'] <= lower_depth)]
organic_carbon = organic_carbon[(organic_carbon['upper_depth'] >= upper_depth) & (organic_carbon['lower_depth'] <= lower_depth)]
pH = pH[(pH['upper_depth'] >= upper_depth) & (pH['lower_depth'] <= lower_depth)]
# water_retention = water_retention[(water_retention['upper_depth'] >= upper_depth) & (water_retention['lower_depth'] <= lower_depth)]
# nitrogen = nitrogen[nitrogen['date'].str.split('-').str[1].str.isnumeric()] # keeps only observations with dates
# organic_carbon = organic_carbon[organic_carbon['date'].str.split('-').str[1].str.isnumeric()] # keeps only observations with dates
# pH = pH[pH['date'].str.split('-').str[1].str.isnumeric()] # keeps only observations with dates
properties = {'nitrogen': {'dataframe': nitrogen, 'title': 'Nitrogen'}, 'organic_carbon': {'dataframe': organic_carbon, 'title': 'Organic carbon'}, 'pH': {'dataframe': pH, 'title': 'pH'}}
TILE_SIZE = utils.read_yaml('config.yml')['TILE_SIZE']

def generate_property_tiles(property_):
    os.makedirs(f'tiles/{property_}', exist_ok=True)
    dataframe = properties[property_]['dataframe']
    dataframe.to_csv(f'{data_dir_path}/{property_}.csv', index=False)
    tiles = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude]).buffer(TILE_SIZE / 2).bounds()).set({'value': measurement.value_avg, 'pos_uncertainty': measurement.positional_uncertainty}) for measurement in dataframe[i: i+5000].itertuples() if measurement.positional_uncertainty == 'Circa 100 m']).getInfo()['features'] for i in range(0, len(dataframe), 5000)])
    tiles = [{**{key: value for key, value in tile.items() if key != 'id'}} for tile in tiles]

    print(len(tiles))
    print('Circa 100 m', len([tile for tile in tiles if tile['properties']['pos_uncertainty'] == 'Circa 100 m']))
    print('100 m - 1 km', len([tile for tile in tiles if tile['properties']['pos_uncertainty'] == '100 m - 1 km']))
    print('1 km - 10 km', len([tile for tile in tiles if tile['properties']['pos_uncertainty'] == '1 km - 10 km']))
    print('Over 10 km', len([tile for tile in tiles if tile['properties']['pos_uncertainty'] == 'Over 10 km']))

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

# print('Nitrogen')
# # nitrogen = nitrogen[nitrogen['date'].str.split('-').str[0].str.isnumeric()] # keeps only observations with dates
# # nitrogen['year'] = nitrogen['date'].str.split('-').str[0].astype(int) # creates a year column
# # nitrogen = nitrogen[nitrogen['year'] >= 2017]
# print('0-5', len(nitrogen[(nitrogen['upper_depth'] >= 0) & (nitrogen['lower_depth'] <= 5)]))

# print('5-15', len(nitrogen[(nitrogen['upper_depth'] >= 5) & (nitrogen['lower_depth'] <= 15)]))

# print('15-30', len(nitrogen[(nitrogen['upper_depth'] >= 15) & (nitrogen['lower_depth'] <= 30)]))

# print('30-60', len(nitrogen[(nitrogen['upper_depth'] >= 30) & (nitrogen['lower_depth'] <= 60)]))

# print('60-100', len(nitrogen[(nitrogen['upper_depth'] >= 60) & (nitrogen['lower_depth'] <= 100)]))

# print('100-200', len(nitrogen[(nitrogen['upper_depth'] >= 100) & (nitrogen['lower_depth'] <= 200)]))


# nitrogen = nitrogen['value_avg']
# print(len(nitrogen))
# print(np.mean(nitrogen.values), np.std(nitrogen.values))
# print(nitrogen.head())
# print(nitrogen['layer_id'].head())
# print(nitrogen[['upper_depth', 'lower_depth']])

# print(layers['layer_number'])
# print(layers[layers['layer_id'] == 606912][['layer_name', 'upper_depth', 'lower_depth', 'layer_number']])
# print(len(nitrogen))
# print(len(nitrogen[nitrogen['year'] >= 2010]))
# profile_id = list(nitrogen['profile_id'])[0]
# site_id = layers[layers['profile_id'] == profile_id]['site_id'].unique()[0]
# print(layers.columns.tolist())
# print(sites.head())

# nitrogen['site_id'] = nitrogen['profile_id'].apply(get_site_id)
# print(len(list(nitrogen['site_id'].unique())))
# nitrogen = nitrogen[nitrogen['site_id'] == list(nitrogen['site_id'].unique())[-3]]
# print(nitrogen.head())
# print(len(nitrogen))

# fig, ax = plt.subplots(dpi=300)
# ax.plot(nitrogen['year'], nitrogen['value_avg'])
# ax.set(xlabel='Year', ylabel='Nitrogen')
# fig.savefig('nitrogen.png')

# nitrogen = nitrogen[nitrogen['year'] >= 2010]

# fig, ax = plt.subplots(dpi=300)
# ax.plot(nitrogen['year'], nitrogen['value_avg'])
# ax.set(xlabel='Year', ylabel='Nitrogen')
# fig.savefig('nitrogen-2010.png')




# print('Organic carbon')
# organic_carbon = organic_carbon[organic_carbon['date'].str.split('-').str[0].str.isnumeric()] # keeps only observations with dates
# # organic_carbon['year'] = organic_carbon['date'].str.split('-').str[0].astype(int) # creates a year column
# # organic_carbon = organic_carbon[organic_carbon['year'] >= 2017]

# print('0-5', len(organic_carbon[(organic_carbon['upper_depth'] >= 0) & (organic_carbon['lower_depth'] <= 5)]))

# print('5-15', len(organic_carbon[(organic_carbon['upper_depth'] >= 5) & (organic_carbon['lower_depth'] <= 15)]))

# print('15-30', len(organic_carbon[(organic_carbon['upper_depth'] >= 15) & (organic_carbon['lower_depth'] <= 30)]))

# print('30-60', len(organic_carbon[(organic_carbon['upper_depth'] >= 30) & (organic_carbon['lower_depth'] <= 60)]))

# print('60-100', len(organic_carbon[(organic_carbon['upper_depth'] >= 60) & (organic_carbon['lower_depth'] <= 100)]))

# print('100-200', len(organic_carbon[(organic_carbon['upper_depth'] >= 100) & (organic_carbon['lower_depth'] <= 200)]))
# organic_carbon = organic_carbon['value_avg']
# print(len(organic_carbon))
# print(np.mean(organic_carbon.values), np.std(organic_carbon.values))

# # print(len(organic_carbon))
# # print(len(organic_carbon[organic_carbon['year'] >= 2010]))

# # fig, ax = plt.subplots(dpi=300)
# # ax.plot(organic_carbon['year'], organic_carbon['value_avg'])
# # ax.set(xlabel='Year', ylabel='Organic carbon')
# # fig.savefig('organic_carbon.png')

# # organic_carbon = organic_carbon[organic_carbon['year'] >= 2010]

# # fig, ax = plt.subplots(dpi=300)
# # ax.plot(organic_carbon['year'], organic_carbon['value_avg'])
# # ax.set(xlabel='Year', ylabel='Organic carbon')
# # fig.savefig('organic_carbon-2010.png')

# print('pH')
# pH = pH[pH['date'].str.split('-').str[0].str.isnumeric()] # keeps only observations with dates
# # pH['year'] = pH['date'].str.split('-').str[0].astype(int) # creates a year column
# # pH = pH[pH['year'] >= 2017]

# print('0-5', len(pH[(pH['upper_depth'] >= 0) & (pH['lower_depth'] <= 5)]))

# print('5-15', len(pH[(pH['upper_depth'] >= 5) & (pH['lower_depth'] <= 15)]))

# print('15-30', len(pH[(pH['upper_depth'] >= 15) & (pH['lower_depth'] <= 30)]))

# print('30-60', len(pH[(pH['upper_depth'] >= 30) & (pH['lower_depth'] <= 60)]))

# print('60-100', len(pH[(pH['upper_depth'] >= 60) & (pH['lower_depth'] <= 100)]))

# print('100-200', len(pH[(pH['upper_depth'] >= 100) & (pH['lower_depth'] <= 200)]))
# pH = pH['value_avg']
# print(len(pH))
# print(np.mean(pH.values), np.std(pH.values))

# # print(len(pH))
# # print(len(pH[pH['year'] >= 2010]))

# # fig, ax = plt.subplots(dpi=300)
# # ax.plot(pH['year'], pH['value_avg'])
# # ax.set(xlabel='Year', ylabel='pH')
# # fig.savefig('pH.png')

# # pH = pH[pH['year'] >= 2010]

# # fig, ax = plt.subplots(dpi=300)
# # ax.plot(pH['year'], pH['value_avg'])
# # ax.set(xlabel='Year', ylabel='pH')
# # fig.savefig('pH-2010.png')

# print('Water retention')
# water_retention = water_retention[water_retention['date'].str.split('-').str[0].str.isnumeric()] # keeps only observations with dates
# # water_retention['year'] = water_retention['date'].str.split('-').str[0].astype(int) # creates a year column
# # water_retention = water_retention[water_retention['year'] >= 2017]

# print('0-5', len(water_retention[(water_retention['upper_depth'] >= 0) & (water_retention['lower_depth'] <= 5)]))

# print('5-15', len(water_retention[(water_retention['upper_depth'] >= 5) & (water_retention['lower_depth'] <= 15)]))

# print('15-30', len(water_retention[(water_retention['upper_depth'] >= 15) & (water_retention['lower_depth'] <= 30)]))

# print('30-60', len(water_retention[(water_retention['upper_depth'] >= 30) & (water_retention['lower_depth'] <= 60)]))

# print('60-100', len(water_retention[(water_retention['upper_depth'] >= 60) & (water_retention['lower_depth'] <= 100)]))

# print('100-200', len(water_retention[(water_retention['upper_depth'] >= 100) & (water_retention['lower_depth'] <= 200)]))
# water_retention = water_retention['value_avg']
# print(len(water_retention))
# print(np.mean(water_retention.values), np.std(water_retention.values))

# # print(len(water_retention))
# # print(len(water_retention[water_retention['year'] >= 2010]))
