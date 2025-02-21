# imports
import ee
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
OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']

def generate_property_points(property_):
    print(properties[property_]['title'])

    dataframe = properties[property_]['dataframe']
    dataframe.to_csv(f'{data_dir_path}/{property_}/{property_}.csv', index=False) # saves the dataframe as a CSV
    points = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude])).set({'value': measurement.value_avg}) for measurement in dataframe[i: i+5000].itertuples() if measurement.positional_uncertainty == 'Circa 100 m']).map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry())).getInfo()['features'] for i in range(0, len(dataframe), 5000)])
    points = [{**{key: value for key, value in point.items() if key != 'id'}} for point in points] # removes ID property

    print(f'{len(points)} points before removing overlaps')

    os.makedirs(f'{property_}/points', exist_ok=True)
    utils.save_geojson(features=points, path=f'{property_}/points/{property_}_points_overlapping.geojson') # saves all the points as a GeoJSON
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path=f'{property_}/points/{property_}_outer_tiles_overlapping.geojson')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    random.shuffle(points) # shuffles the points list
    points = [{**point, 'id': i} for i, point in enumerate(points)] # assigns each point an ID
    utils.save_geojson(features=points, path=f'{property_}/points/{property_}_points.geojson') # saves the points as a GeoJSON
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path=f'{property_}/points/{property_}_outer_tiles.geojson')

    # os.makedirs(f'{property_}/figures', exist_ok=True)
    # utils.make_global_map(tiles=tiles, color=properties[property_]['color'], path=f'{property_}/figures/{property_}_map', title=properties[property_]['title'])

for property_ in properties.keys():
    generate_property_points(property_)
