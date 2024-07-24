'''
get_tile_data.py by Lucia Gordon and Vishal Nedungadi
'''

# imports
from ee_data import EEData
from utils import read_geojson, read_yaml
import ee
import geojson
import json
import numpy as np
import os
import pandas as pd

ee.Initialize(project='mmearth-bench') # initializes EE with our project

def process_tile(tile):
    year = '2020'
    collection_names = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                            .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"') # get feature collections with features in the selected year
                            .filterBounds(tile['geometry']) # get feature collections that have features within the tile
                            .aggregate_array('table_id') # extract the IDs of the feature collections
                            .getInfo()) # list of names of the feature collections
    quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
    points = (ee.FeatureCollection([ee.FeatureCollection(name) for name in collection_names])
                .flatten() # merge all the feature collections into one
                .filterBounds(tile['geometry']) # collection of the features that are within the tile
                .filter(quality_filter) # apply the quality filter
                .map(lambda point: point.set('off_after_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
                .filter(ee.Filter.gt('off_after_on', 0))) # filtering by points with leaf on before leaf off
    leaf_on_off = np.array([points.aggregate_array('leaf_on_doy').getInfo(), points.aggregate_array('leaf_off_doy').getInfo()]).T # get pairs of leaf on and off days for each point
    leaf_on_off_unique = list(map(list, set(map(tuple, leaf_on_off)))) # get unique pairs
    leaf_on_off_dates = [[pd.to_datetime(pair[0], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d'), pd.to_datetime(pair[1], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d')] for pair in leaf_on_off_unique]

    return leaf_on_off_dates, points

def main():
    task = 'biomass'
    os.makedirs(f'tiles/{task}/pixel_level_data', exist_ok=True)
    gj = read_geojson(f'tiles/{task}/{task}_tiles.geojson') # reading the GeoJSON file
    tile_image_level_data = {}
    start_tile = 0
    end_tile = len(gj['features'])
    tiles_made = 0

    for tile_index in range(start_tile, end_tile):
        print(f'Processing tile {tile_index+1}/{end_tile}')
        tile = gj['features'][tile_index]
        leaf_on_off_dates, points = process_tile(tile)

        if len(leaf_on_off_dates) > 0:
            ee_data = EEData(tile, task, leaf_on_off_dates, points)

            if not ee_data.no_data:
                tiles_made += 1
                tile_image_level_data[tile['id']] = {'biome': ee_data.biome,
                                                     'ecoregion': ee_data.ecoregion,
                                                     'era5': ee_data.era5_data,
                                                     's2_date': ee_data.s2_date,
                                                     'geolocation_encoding': ee_data.geolocation_encoding,
                                                     'month_encoding': ee_data.month_encoding,
                                                     'crs': ee_data.crs,
                                                     'lat': ee_data.lat,
                                                     'lon': ee_data.lon}

    with open(f'tiles/{task}/tile_image_level_data.json', 'w') as file:
        json.dump(tile_image_level_data, file, indent=4)

    print(f'{tiles_made} tiles made')

if __name__ == '__main__':
    main()
