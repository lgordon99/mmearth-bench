'''
generate_species_tiles.py by Lucia Gordon
Need 30GB RAM to run this file
'''

# imports
from sys import argv
from utils import read_yaml, read_json, get_biomes_ecoregions_area
import ee
import json
import math
import os
import pandas as pd

ee.Initialize(project='mmearth-bench') # initializes EE with our project
ecoregions_collection = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
title = 'biomes_ecoregions.json'
df = pd.read_csv('/n/tambe_lab/Users/luciagordon/sinr-data/train/geo_prior_train.csv')
df['year'] = df['observed_on'].str.split('-').str[0].astype(int)
df['month'] = df['observed_on'].str.split('-').str[1].astype(int)

print(f'Columns = {df.columns.tolist()}')
print(f'{len(df)} observations')
print(f'{df["taxon_id"].nunique()} species')

# observations_post_2017 = df[df['year'] >= 2017]

# print(f'{len(observations_post_2017)} observations between 2017-2021')
# print(f'{observations_post_2017["taxon_id"].nunique()} species between 2017-2021')

observations_2020 = df[df['year'] == 2020]

print(f'{len(observations_2020)} observations in 2020')
print(f'{observations_2020["taxon_id"].nunique()} species in 2020')

species_observation_collection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([observation.longitude, observation.latitude]), {'species': observation.taxon_id, 'month': observation.month}) for observation in observations_2020.itertuples()])

# def make_species_observation_collection():
#     df = pd.read_csv('/n/tambe_lab/Users/luciagordon/sinr-data/train/geo_prior_train.csv')
#     df['year'] = df['observed_on'].str.split('-').str[0].astype(int)
#     df['month'] = df['observed_on'].str.split('-').str[1].astype(int)

#     print(f'Columns = {df.columns.tolist()}')
#     print(f'{len(df)} observations')
#     print(f'{df["taxon_id"].nunique()} species')

#     observations_post_2017 = df[df['year'] >= 2017]

#     print(f'{len(observations_post_2017)} observations between 2017-2021')
#     print(f'{observations_post_2017["taxon_id"].nunique()} species between 2017-2021')

#     observations_2020 = df[df['year'] == 2020]

#     print(f'{len(observations_2020)} observations in 2020')
#     print(f'{observations_2020["taxon_id"].nunique()} species in 2020')

#     species_observation_collection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([observation.longitude, observation.latitude]), {'species': observation.taxon_id, 'month': observation.month}) for observation in observations_2020.itertuples()])

#     with open('species_observation_collection.geojson', 'w') as file:
#         json.dump(species_observation_collection.getInfo(), file, indent=4)

def generate_species_tiles():
    NUM_TILES = 100
    TILE_SIZE = read_yaml('config.yml')['TILE_SIZE']

    if not os.path.exists(f'biomes_ecoregions_data/{title}'):
        get_biomes_ecoregions_area(ecoregions_collection=ecoregions_collection, title=title)

    biomes_ecoregions = read_json(f'biomes_ecoregions_data/{title}')
    biomes = biomes_ecoregions.keys()
    num_tiles_per_biome = math.ceil(NUM_TILES / len(biomes))
    tiles = []
    os.makedirs('tiles', exist_ok=True)

    print(f'{len(biomes)} biomes')
    print(f'{num_tiles_per_biome} tiles per biome')

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')

        ecoregions = biomes_ecoregions[biome]['ecoregions'].keys()
    
        for j, ecoregion in enumerate(ecoregions):
            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')

            ecoregion_collection = ecoregions_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
            ecoregion_points = species_observation_collection.filterBounds(ecoregion_collection)
            # num_ecoregion_tiles = min(ecoregion_points.size().getInfo(), math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area'])) # number of tiles per ecoregion is proportional to the size of the ecoregion
            num_ecoregion_tiles = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion

            print(f'{num_ecoregion_tiles} tiles in ecoregion')

            while num_ecoregion_tiles > 0:
                candidate_tile = ee.FeatureCollection.randomPoints(region=ecoregion_points, points=1).map(lambda point: point.buffer(TILE_SIZE / 2).bounds()).first().getInfo()

                # for tile in tiles:
                #     if ee.Geometry.Polygon(candidate_tile['geometry']['coordinates']).intersects(ee.Geometry.Polygon(tile['geometry']['coordinates']), maxError=1).getInfo(): # if the candidate tile overlaps with any existing tiles
                #         print('Overlap detected')
                #         continue

                candidate_tile['id'] = len(tiles)
                candidate_tile['properties'] = {'biome': biome, 'ecoregion': ecoregion}
                print(candidate_tile)
                tiles.append(candidate_tile)
                num_ecoregion_tiles -= 1
                print(f'{len(tiles)} tiles total')

                random.shuffle(tiles) # shuffling the tiles list
                geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

                with open('tiles/species_tiles.geojson', 'w') as f:
                    json.dump(geojson_collection, f, indent=4)

if __name__ == '__main__':
    generate_species_tiles()
