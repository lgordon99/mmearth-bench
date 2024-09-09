'''
generate_species_tiles.py by Lucia Gordon
Need 100GB? RAM to run this file
'''

# imports
from geopy.distance import geodesic
from pyproj import Geod
from shapely.geometry import mapping, MultiPoint, Point, Polygon
from sys import argv
from utils import read_yaml, read_json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import geemap
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
        if 'iconic_taxon_name' in list(response.json()['results'][0].keys()):
            return response.json()['results'][0]['iconic_taxon_name']
        else:
            return None
    else:
        get_species_type(species)

def calculate_spread(observations):
    observations = observations.values.tolist()
    geod = Geod(ellps='WGS84')
    area, _ = geod.geometry_area_perimeter(MultiPoint(observations).convex_hull)

    return abs(area)

def generate_species_tiles():
    observations_2020 = pd.read_csv(f'{data_dir_path}/sinr-data/observations_2020.csv', usecols=['latitude', 'longitude', 'taxon_id', 'month']) # reads in the 2020 data
    num_observations = len(observations_2020) # gets the total number of observations
    metadata = read_json(f'{data_dir_path}/sinr-data/train/geo_prior_train_meta.json') # reads in metadata
    species_id_name = {item['taxon_id']: item['latin_name'] for item in metadata} # maps each taxon ID to a species name
    observations_2020['species'] = observations_2020['taxon_id'].map(species_id_name) # creates a species column
    species_count = observations_2020['species'].value_counts().reset_index() # number of observations for each species
    species_count = species_count[species_count['count'] >= 300] # keeps species with at least 300 observations
    species_count_300 = species_count['species'].tolist()
    print(f'{len(species_count_300)} species with at least 300 observations')

    observations_species_count_300 = observations_2020[observations_2020['species'].isin(species_count_300)] # observations for species with at least 300 observations
    species_spread = observations_species_count_300.groupby('species')[['longitude', 'latitude']].apply(calculate_spread).reset_index(name='spread')
    species_count = pd.merge(species_count, species_spread[['species', 'spread']], on='species', how='left') # adds spread column
    species_count = species_count.sort_values(by='spread', ascending=False).head(300) # takes the top species in terms of spread
    species_count = species_count.sort_values(by='count', ascending=False) # sorts the species in descending order by count
    species_types = {species: result for species, result in ((species, get_species_type(species)) for species in species_count['species']) if result is not None}
    species_count = species_count[species_count['species'].isin(list(species_types.keys()))] # selects only the species that had type data
    species_count['type'] = species_count['species'].map(species_types) # creates a type column
    type_counts = species_count['type'].value_counts().reset_index() # counts the occurrences of each type
    print(f'{len(list(species_types.keys()))} species with top geographic spread')

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
        tile['geometry'] = Polygon(tile['geometry']['coordinates'][0])
        month = tile['properties']['month']
        previous_month = month - 1 if month > 1 else 12
        next_month = month + 1 if month < 12 else 1
        observations_in_tile = observations_gdf[observations_gdf.within(tile['geometry'])]
        filtered_data = observations_in_tile[(observations_in_tile['month'] >= previous_month) & (observations_in_tile['month'] <= next_month)]
        tile['properties']['species'] = list(filtered_data['species'].unique())

    print(f'{len(tiles)} tiles before removing overlaps')

    geojson_collection = {'type': 'FeatureCollection', 'features': [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': mapping(tile['geometry'])} for tile in tiles]}

    with open('tiles/species/species_tiles_all.geojson', 'w') as f:
        json.dump(geojson_collection, f, indent=4) # save the tiles as a GeoJSON

    tiles_gdf = gpd.GeoDataFrame(tiles, geometry='geometry').reset_index(drop=False)
    intersections = gpd.sjoin(tiles_gdf, tiles_gdf)
    intersecting_indices = intersections[intersections['index_left'] != intersections['index_right']][['index_left', 'index_right']].to_numpy().tolist()
    indices_to_remove = []

    while len(intersecting_indices) > 0:
        index_to_remove = intersecting_indices[0][0]
        indices_to_remove.append(index_to_remove)
        intersecting_indices = [pair for pair in intersecting_indices if index_to_remove not in pair]

    indices_to_keep = [i for i in range(len(tiles)) if i not in indices_to_remove]
    tiles = [tiles[i] for i in indices_to_keep]

    print(f'{len(tiles)} tiles')

    os.makedirs('tiles/species', exist_ok=True)
    random.shuffle(tiles) # shuffling the tiles list
    geojson_collection = {'type': 'FeatureCollection', 'features': [{**{key: value for key, value in tile.items() if key != 'geometry'}, 'geometry': mapping(tile['geometry'])} for tile in tiles]}

    with open('tiles/species/species_tiles.geojson', 'w') as f:
        json.dump(geojson_collection, f, indent=4) # save the tiles as a GeoJSON

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

if __name__ == '__main__':
    if not os.path.exists(f'{data_dir_path}/sinr-data/observations_2020.csv'):
        save_2020_observations()

    generate_species_tiles()
