# imports
from sys import argv
from utils import read_yaml, read_json
import argparse
import ee
import geemap
import geopandas as gpd
import json
import math
import numpy as np
import os
import pandas as pd
import random
import subprocess
import time

def generate_ecoregion_tiles(biome, ecoregion):
    start_time = time.time()
    ee.Initialize(project='mmearth-bench') # initializes EE with our project
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
    NUM_TILES = 1000
    TILE_SIZE = read_yaml('config.yml')['TILE_SIZE']
    year = '2020'
    gedi_feature_collection = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                                 .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"')) # get feature collections with features in the selected year
    quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
    biomes_ecoregions = read_json(f'biomes_ecoregions_data/biomes_ecoregions_gedi.json')
    biomes = biomes_ecoregions.keys()
    num_tiles_per_biome = math.ceil(NUM_TILES / len(biomes))
    tiles = []
    os.makedirs('tiles/biomass/ecoregion_tiles', exist_ok=True)

    print(f'Ecoregion: {ecoregion}')

    ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
    collection_names = (gedi_feature_collection
                        .filterBounds(ecoregion_collection) # get feature collections that have features within the ecoregion
                        .aggregate_array('table_id') # extract the IDs of the feature collections
                        .getInfo()) # list of names of the feature collections
    ecoregion_points = (ee.FeatureCollection([ee.FeatureCollection(name) for name in collection_names])
                          .flatten() # merge all the feature collections into one
                          .filterBounds(ecoregion_collection) # collection of the features that are within the ecoregion
                          .filter(quality_filter))
    num_ecoregion_tiles = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion

    print(f'{num_ecoregion_tiles} tiles in ecoregion')

    while num_ecoregion_tiles > 0:
        candidate_tiles = ee.FeatureCollection.randomPoints(region=ecoregion_points, points=min(num_ecoregion_tiles, 5000)).map(lambda point: point.buffer(TILE_SIZE / 2).bounds())
        candidate_tiles_list = candidate_tiles.toList(candidate_tiles.size())

        # adding candidate tile
        for i in range(candidate_tiles.size().getInfo()):
            candidate_tile = candidate_tiles_list.get(i).getInfo()
            candidate_tile['id'] = len(tiles) # set the ID to be the number of tiles that have been made so far
            candidate_tile['properties']['biome'] = biome
            candidate_tile['properties']['ecoregion'] = ecoregion # save the biome and ecoregion for the tile
            tiles.append(candidate_tile) # add the tile to the list
            num_ecoregion_tiles -= 1 # subtract 1 from the number of tiles left to be made for the ecoregion

    # random.shuffle(tiles) # shuffling the tiles list
    geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    with open(f'tiles/biomass/ecoregion_tiles/biome_{biome}_ecoregion_{ecoregion.replace('/', '_')}_biomass_tiles.geojson', 'w') as f:
        json.dump(geojson_collection, f, indent=4) # save the tiles as a GeoJSON

    seconds = time.time() - start_time
    print(f'{seconds // 60}:{seconds % 60}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--biome', type=str)
    parser.add_argument('--ecoregion', type=str)
    args = parser.parse_args()

    generate_ecoregion_tiles(args.biome.replace('_', ' '), args.ecoregion.replace('_', ' '))
