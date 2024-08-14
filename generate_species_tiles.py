'''
generate_species_tiles.py by Lucia Gordon
Need 100GB? RAM to run this file
'''

# imports
from sys import argv
from utils import read_yaml, read_json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import json
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests

ee.Initialize(project='mmearth-bench') # initializes EE with our project
os.makedirs('figures', exist_ok=True)
data_dir_path = read_yaml('config-user.yml')['data_dir_path']

def save_2020_observations():
    all_observations_df = pd.read_csv(f'{data_dir_path}/sinr-data/train/geo_prior_train.csv') # converts the CSV to a dataframe
    all_observations_df['year'] = all_observations_df['observed_on'].str.split('-').str[0].astype(int) # creates a year column
    all_observations_df['month'] = all_observations_df['observed_on'].str.split('-').str[1].astype(int) # creates a month column

    observations_2020 = all_observations_df[all_observations_df['year'] == 2020] # extracts the observations from 2020
    observations_2020.to_csv(f'{data_dir_path}/sinr-data/observations_2020.csv', index=False) # saves the 2020 observations as a CSV

    print(f'Columns = {all_observations_df.columns.tolist()}')
    print(f'{len(all_observations_df)} observations')
    print(f'{all_observations_df["taxon_id"].nunique()} species')
    print(f'{len(observations_2020)} observations in 2020')
    print(f'{observations_2020["taxon_id"].nunique()} species in 2020')

def get_species_type(species_name):
    return requests.get(f'https://api.inaturalist.org/v1/taxa?q={species_name}').json()['results'][0]['iconic_taxon_name']

def plot_num_observations_per_species():
    observations_2020 = pd.read_csv(f'{data_dir_path}/sinr-data/observations_2020.csv', usecols=['latitude', 'longitude', 'taxon_id', 'month']) # reads in the 2020 data
    num_observations = len(observations_2020) # gets the total number of observations
    metadata = read_json(f'{data_dir_path}/sinr-data/train/geo_prior_train_meta.json') # reads in metadata
    species_id_name = {item['taxon_id']: item['latin_name'] for item in metadata} # maps each taxon ID to a species name
    observations_2020['species'] = observations_2020['taxon_id'].map(species_id_name) # creates a species column
    num_observations_per_species = observations_2020['species'].value_counts().reset_index() # number of observations for each species
    num_observations_per_species = num_observations_per_species[num_observations_per_species['count'] >= 300] # keeps species with at least 300 observations
    species_spread = observations_2020.groupby('species')[['latitude', 'longitude']].std().reset_index() # calculates STD of latitude and longitude
    species_spread['spread'] = (species_spread['latitude'] ** 2 + species_spread['longitude'] ** 2) ** 0.5 # calculates sqrt(lat_std^2 + lon_std^2)
    num_observations_per_species = pd.merge(num_observations_per_species, species_spread[['species', 'spread']], on='species', how='left') # adds spread column
    num_observations_per_species = num_observations_per_species.sort_values(by='spread', ascending=False).head(100) # takes the top 100 species in terms of spread
    num_observations_per_species = num_observations_per_species.sort_values(by='count', ascending=False) # sorts the species in descending order by count
    species_types = {species: get_species_type(species) for species in num_observations_per_species['species']} 
    num_observations_per_species['type'] = num_observations_per_species['species'].map(species_types)
    type_counts = num_observations_per_species['type'].value_counts().reset_index()

    fig, ax = plt.subplots(dpi=300)
    ax.bar(num_observations_per_species['species'], num_observations_per_species['count'])
    ax.set_ylabel('Count')
    plt.xticks(rotation=90, fontsize=4)
    plt.savefig('figures/count_per_species.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig('figures/count_per_species.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

    fig, ax = plt.subplots(dpi=300)
    ax.bar(type_counts['type'], type_counts['count'])
    ax.set_ylabel('Count')
    plt.xticks(rotation=90)
    plt.savefig('figures/count_per_type.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig('figures/count_per_type.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)

    selected_species = num_observations_per_species['species'].unique() # gets species selected by above filtering procedure
    observations_selected_species = observations_2020[observations_2020['species'].isin(selected_species)][['longitude', 'latitude', 'species']].values.tolist() # gets all observations for the selected species
    observations_dict = {species: [[observation[0], observation[1]] for observation in observations_selected_species if observation[2] == species] for species in selected_species}

    for lon, lat, _ in observations_selected_species:
        plt.plot(lon, lat, marker='o', color='red', markersize=1, transform=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    plt.savefig('map.pdf', bbox_inches='tight')
    plt.savefig('map.png', bbox_inches='tight')

    print(f'{num_observations} observations')
    print(f'{len(num_observations_per_species)} species')
    print(len(species_spread))
    print(num_observations_per_species.columns.tolist())
    print(num_observations_per_species.head())

def generate_species_tiles():
    TILE_SIZE = read_yaml('config.yml')['TILE_SIZE']
    tile = ee.Feature(ee.Geometry.Point([lon, lat])).map(lambda point: point.buffer(TILE_SIZE / 2).bounds())
    tiles = [{**{key: value for key, value in tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for tile in tiles.getInfo()['features']]
    # TODO: get coordinates from observations_dict to make tiles
# ecoregions_collection = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
# title = 'biomes_ecoregions.json'

# species = observations_2020['taxon_id'].unique()
# num_observations_per_species = observations_2020

# num_observations_per_species = {species_: len(observations_2020[observations_2020['taxon_id'] == species_]) for species_ in species}
# num_observations_per_species = dict(sorted(num_observations_per_species.items(), key=lambda item: item[1], reverse=True))

# print(num_observations_per_species[:10])

# fig, ax = plt.subplots(dpi=300)
# ax.bar(species, num_observations_per_species.values())
# ax.set_ylabel('# Observations')
# plt.savefig('figures/num_observations_per_species.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
# plt.savefig('figures/num_observations_per_species_results.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

# species_observation_collection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([observation.longitude, observation.latitude]), {'species': observation.taxon_id, 'month': observation.month}) for observation in observations_2020.itertuples()])

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
    NUM_TILES = 1000

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
            # ecoregion_points = species_observation_collection.filterBounds(ecoregion_collection)
            # num_ecoregion_tiles = min(ecoregion_points.size().getInfo(), math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area'])) # number of tiles per ecoregion is proportional to the size of the ecoregion
            num_ecoregion_tiles = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion

            print(f'{num_ecoregion_tiles} tiles in ecoregion')
            TILE_SIZE = read_yaml('config.yml')['TILE_SIZE']
            tiles = (species_observation_collection
                     .filterBounds(ecoregion_collection)
                     .map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                     .randomColumn('random') # add a new property to each feature containing a random number
                     .sort('random') # sort the features by the random numbers
                     .limit(num_ecoregion_tiles) # select the first points
                     .map(lambda point: point.buffer(TILE_SIZE / 2).bounds())) # convert points to tiles
            tiles = [{**{key: value for key, value in tile.items() if key != 'id'}, 'properties': {'biome': biome, 'ecoregion': ecoregion}} for tile in tiles.getInfo()['features']]
            print(f'{len(tiles)} tiles made')

            # while num_ecoregion_tiles > 0:
            #     candidate_tile = ee.FeatureCollection.randomPoints(region=ecoregion_points, points=1).map(lambda point: point.buffer(TILE_SIZE / 2).bounds()).first().getInfo()

            #     # for tile in tiles:
            #     #     if ee.Geometry.Polygon(candidate_tile['geometry']['coordinates']).intersects(ee.Geometry.Polygon(tile['geometry']['coordinates']), maxError=1).getInfo(): # if the candidate tile overlaps with any existing tiles
            #     #         print('Overlap detected')
            #     #         continue

            #     candidate_tile['id'] = len(tiles)
            #     candidate_tile['properties'] = {'biome': biome, 'ecoregion': ecoregion}
            #     print(candidate_tile)
            #     tiles.append(candidate_tile)
            #     num_ecoregion_tiles -= 1
            #     print(f'{len(tiles)} tiles total')

            #     random.shuffle(tiles) # shuffling the tiles list
            geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

            with open(f'tiles/species/species_tiles/biome_{biome}_ecoregion_{ecoregion.replace("/", "_")}_biomass_tiles.geojson', 'w') as f:
                json.dump(geojson_collection, f, indent=4)

if __name__ == '__main__':
    if not os.path.exists(f'{data_dir_path}/sinr-data/observations_2020.csv'):
        save_2020_observations()

    plot_num_observations_per_species()
    # generate_species_tiles()
