'''
generate_species_tiles.py by Lucia Gordon
Need 100GB? RAM to run this file
'''

# imports
from geopy.distance import geodesic
from pyproj import Geod
from shapely.geometry import MultiPoint, Point, Polygon
from sys import argv
from utils import read_yaml, read_json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import geopandas as gpd
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
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

def get_species_type(species):
    response = requests.get(f'https://api.inaturalist.org/v1/taxa?q={species}')

    if response.status_code == 200:
        data = response.json()['results'][0]['iconic_taxon_name']
        return data
    else:
        get_species_type(species)

def calculate_spread(observations):
    observations = observations.values.tolist()
    # distances = []

    # for i in range(len(observations)):
    #     for j in range(len(observations)):
    #         if i != j:
    #             distances.append(geodesic(observations[i], observations[j]).kilometers)

    # return np.mean(distances)


    # Use Geod to calculate the area of the convex hull on Earth's surface
    geod = Geod(ellps='WGS84')
    area, _ = geod.geometry_area_perimeter(MultiPoint(observations).convex_hull)

    return abs(area)

def plot_species_count():
    observations_2020 = pd.read_csv(f'{data_dir_path}/sinr-data/observations_2020.csv', usecols=['latitude', 'longitude', 'taxon_id', 'month']) # reads in the 2020 data
    num_observations = len(observations_2020) # gets the total number of observations
    metadata = read_json(f'{data_dir_path}/sinr-data/train/geo_prior_train_meta.json') # reads in metadata
    species_id_name = {item['taxon_id']: item['latin_name'] for item in metadata} # maps each taxon ID to a species name
    observations_2020['species'] = observations_2020['taxon_id'].map(species_id_name) # creates a species column
    species_count = observations_2020['species'].value_counts().reset_index() # number of observations for each species
    species_count = species_count[species_count['count'] >= 300] # keeps species with at least 300 observations
    species_count_300 = species_count['species'].tolist()
    observations_species_count_300 = observations_2020[observations_2020['species'].isin(species_count_300)]
    species_spread = observations_species_count_300.groupby('species')[['longitude', 'latitude']].apply(calculate_spread).reset_index(name='spread')
    print(species_spread.head())
    # species_spread = observations_2020.groupby('species')[['latitude', 'longitude']].std().reset_index() # calculates STD of latitude and longitude
    # species_spread['spread'] = (species_spread['latitude'] ** 2 + species_spread['longitude'] ** 2) ** 0.5 # calculates sqrt(lat_std^2 + lon_std^2)
    species_count = pd.merge(species_count, species_spread[['species', 'spread']], on='species', how='left') # adds spread column
    species_count = species_count.sort_values(by='spread', ascending=False).head(100) # takes the top 100 species in terms of spread
    species_count = species_count.sort_values(by='count', ascending=False) # sorts the species in descending order by count
    species_types = {species: requests.get(f'https://api.inaturalist.org/v1/taxa?q={species}').json()['results'][0]['iconic_taxon_name'] for species in species_count['species']} 
    species_count['type'] = species_count['species'].map(species_types) # creates a type column
    type_counts = species_count['type'].value_counts().reset_index() # counts the occurrences of each type

    # plot count per species
    fig, ax = plt.subplots(dpi=300)
    ax.bar(species_count['species'], species_count['count'])
    ax.set_ylabel('Count')
    plt.xticks(rotation=90, fontsize=4)
    plt.savefig('figures/count_per_species.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig('figures/count_per_species.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

    # plot count per type
    fig, ax = plt.subplots(dpi=300)
    ax.bar(type_counts['type'], type_counts['count'])
    ax.set_ylabel('Count')
    plt.xticks(rotation=90)
    plt.savefig('figures/count_per_type.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig('figures/count_per_type.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

    selected_species = species_count['species'].tolist() # gets species selected by above filtering procedure
    observations_selected_species = observations_2020[observations_2020['species'].isin(selected_species)][['species', 'month', 'longitude', 'latitude']].values.tolist() # gets all observations for the selected species
    observations_dict = {species: [[observation[1], observation[2], observation[3]] for observation in observations_selected_species if observation[0] == species] for species in selected_species} # groups observations by species
    selected_observations_dict = {species: random.sample(observations_dict[species], 300) for species in selected_species} # samples 300 observations per species
    observations_selected_species_df = observations_2020[observations_2020['species'].isin(selected_species)][['species', 'month', 'longitude', 'latitude']] # gets all observations for the selected species
    observations_selected_species_df['geometry'] = observations_selected_species_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1) # adds a geometry column
    observations_gdf = gpd.GeoDataFrame(observations_selected_species_df, geometry='geometry') # creates a GeoDataFrame for all the observations for the selected species

    print(np.array(list(selected_observations_dict.values())).shape)

    TILE_SIZE = read_yaml('config.yml')['TILE_SIZE']
    tiles = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([observation[1], observation[2]]).buffer(TILE_SIZE / 2).bounds()).set({'species': species, 'month': observation[0]}) for observation in selected_observations_dict[species]]).getInfo()['features'] for species in selected_species])
    tiles = [{**{key: value for key, value in tile.items() if key != 'id'}} for tile in tiles]

    for tile in tiles:
        month = tile['properties']['month']
        previous_month = month - 1 if month > 1 else 12
        next_month = month + 1 if month < 12 else 1
        observations_in_tile = observations_gdf[observations_gdf.within(Polygon(tile['geometry']['coordinates'][0]))]
        filtered_data = observations_in_tile[(observations_in_tile['month'] >= previous_month) & (observations_in_tile['month'] <= next_month)]
        tile['properties']['species'] = list(filtered_data['species'].unique())

    print(f'{len(tiles)} tiles')
    os.makedirs('tiles/species', exist_ok=True)

    geojson_collection = {'type': 'FeatureCollection', 'features': tiles}

    with open('tiles/species/species_tiles.geojson', 'w') as f:
        json.dump(geojson_collection, f, indent=4) # save the tiles as a GeoJSON

    tiles_list = read_geojson('tiles/species/species_tiles.geojson')['features']
    print(len(tiles_list))

    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)

    selected_observations = np.array(list(selected_observations_dict.values()))[:, :, 1:].reshape(-1, 2)

    for observation in selected_observations:
        lon, lat = observation
        plt.plot(lon, lat, marker='o', color='red', markersize=0.5, transform=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    plt.savefig('map.pdf', bbox_inches='tight')
    plt.savefig('map.png', bbox_inches='tight')

    # print(f'{num_observations} observations')
    # print(f'{len(species_count)} species')
    # print(len(species_spread))
    # print(species_count.columns.tolist())
    # print(species_count.head())

# def generate_species_tiles():
# ecoregions_collection = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
# title = 'biomes_ecoregions.json'

# species = observations_2020['taxon_id'].unique()
# species_count = observations_2020

# species_count = {species_: len(observations_2020[observations_2020['taxon_id'] == species_]) for species_ in species}
# species_count = dict(sorted(species_count.items(), key=lambda item: item[1], reverse=True))

# print(species_count[:10])

# fig, ax = plt.subplots(dpi=300)
# ax.bar(species, species_count.values())
# ax.set_ylabel('# Observations')
# plt.savefig('figures/species_count.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
# plt.savefig('figures/species_count_results.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

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

    plot_species_count()
    # generate_species_tiles()
