# imports
import ee
import numpy as np
import pandas as pd
import random
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
random.seed(42)

# read in data
nitrogen = pd.read_csv(f'{data_dir_path}/soil_nitrogen/wosis_202312_nitkjd.tsv', sep='\t')
organic_carbon = pd.read_csv(f'{data_dir_path}/soil_organic_carbon/wosis_202312_orgc.tsv', sep='\t', low_memory=False)
pH = pd.read_csv(f'{data_dir_path}/soil_pH/wosis_202312_phaq.tsv', sep='\t')

properties = {'soil_nitrogen': {'dataframe': nitrogen, 'title': 'Soil nitrogen', 'color': 'b'},
              'soil_organic_carbon': {'dataframe': organic_carbon, 'title': 'Soil organic carbon', 'color': 'tab:brown'},
              'soil_pH': {'dataframe': pH, 'title': 'Soil pH', 'color': 'tab:purple'}}

# depth range
upper_depth = 0 # cm
lower_depth = 5 # cm

OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']

def generate_property_points(property_):
    print(properties[property_]['title'])

    dataframe = properties[property_]['dataframe']
    dataframe = dataframe[(dataframe['upper_depth'] >= upper_depth) & (dataframe['lower_depth'] <= lower_depth) & (dataframe['positional_uncertainty'] == 'Circa 100 m')] # filters for measurements in selected depth range and with the lowest positional uncertainty
    points = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([measurement.longitude, measurement.latitude])).set({property_: measurement.value_avg}) for measurement in dataframe[i: i+5000].itertuples()]).map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry())).getInfo()['features'] for i in range(0, len(dataframe), 5000)])
    points = [{**{key: value for key, value in point.items() if key != 'id'}} for point in points] # removes ID property
    print(f'{len(points)} points before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    random.shuffle(points) # shuffles the points list
    points = [{**{key: value for key, value in point.items() if key != 'properties'}, 'properties': {key: value for key, value in point['properties'].items() if key != 'outer_tile'}, 'id': i} for i, point in enumerate(points)] # assigns each point an ID and removes the outer tile
    utils.save_geojson(features=points, path=f'{data_dir_path}/{property_}/{property_}_points.geojson') # saves the points as a GeoJSON

for property_ in properties.keys():
    generate_property_points(property_)
