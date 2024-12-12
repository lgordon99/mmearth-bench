'''
generate_species_points.py by Lucia Gordon
'''

# imports
from bs4 import BeautifulSoup
from PIL import Image
from pyproj import Geod
from shapely.geometry import MultiPoint, Point
from sys import argv
import csv
import ee
import geopandas as gpd
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import requests
import subprocess
import time
import utils

ee.Initialize(project='mmearth-bench') # initializes EE with our project
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
final_num_tiles_per_species = 200 # in the end each species is the "main species" in 200 tiles

def save_2020_observations():
    '''requires 20 GB of memory'''

    start_time = time.time()
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
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

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
    # observations = observations.values.tolist()
    geod = Geod(ellps='WGS84')
    area, _ = geod.geometry_area_perimeter(MultiPoint(observations.values.tolist()).convex_hull) # area of convex hull for the points

    return abs(area)

def get_all_species_observation_counts(points, selected_species):
    species_observations_counts = {species: 0 for species in selected_species}

    for point in points:
        species_observations_counts[point['properties']['main_species']] += 1

        # if isinstance(point['properties']['species'], str): # if there is only one species
        #     species_observations_counts[point['properties']['species']] += 1
        # else: # if there are potentially multiple species
        #     for species in point['properties']['species']:
        #         species_observations_counts[species] += 1

    return species_observations_counts

def generate_species_points():
    start_time = time.time()

    os.makedirs('species/points', exist_ok=True)
    os.makedirs('species/figures', exist_ok=True)

    # collect species with some minimum number of observations
    observations_2020 = pd.read_csv(f'{data_dir_path}/sinr-data/observations_2020.csv', usecols=['latitude', 'longitude', 'taxon_id', 'month']) # reads in the 2020 data
    num_observations = len(observations_2020) # gets the total number of observations
    metadata = utils.read_json(f'{data_dir_path}/sinr-data/train/geo_prior_train_meta.json') # reads in metadata
    species_id_name = {item['taxon_id']: item['latin_name'] for item in metadata} # maps each taxon ID to a species name
    observations_2020['species'] = observations_2020['taxon_id'].map(species_id_name) # creates a species column
    species_count = observations_2020['species'].value_counts().reset_index() # number of observations for each species
    min_num_observations_per_species = 400
    species_count = species_count[species_count['count'] >= min_num_observations_per_species] # only keeps species with some minimum number of observations
    species_enough_observations = species_count['species'].tolist()
    print(f'{len(species_enough_observations)} species with at least {min_num_observations_per_species} observations')

    # select species with the highest geographic spread in their observations
    observations_species_enough_observations = observations_2020[observations_2020['species'].isin(species_enough_observations)] # observations for species with some minimum number of observations
    species_spread = observations_species_enough_observations.groupby('species')[['longitude', 'latitude']].apply(calculate_spread).reset_index(name='spread') # calculates geographic spread of each species' observations
    species_count = pd.merge(species_count, species_spread[['species', 'spread']], on='species', how='left') # adds spread column
    species_count = species_count.sort_values(by='spread', ascending=False).head(400) # takes the top species in terms of spread
    species_count = species_count.sort_values(by='count', ascending=False) # sorts the species in descending order by count
    selected_species = species_count['species'].tolist() # gets species selected by above filtering procedure
    print(f'{len(selected_species)} species with top geographic spread')

    # plot count per species
    fig, ax = plt.subplots(dpi=300)
    ax.bar(species_count['species'], species_count['count'])
    ax.set_ylabel('Count')
    plt.xticks(rotation=90, fontsize=4)
    plt.savefig('figures/count_per_species.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.savefig('figures/count_per_species.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

    # randomly select a fixed number of observations for each species
    observations_selected_species_df = observations_2020[observations_2020['species'].isin(selected_species)][['species', 'month', 'longitude', 'latitude']] # gets all observations for the selected species
    observations_selected_species_dict = {species: [[observation.month, observation.longitude, observation.latitude] for observation in observations_selected_species_df.itertuples() if observation.species == species] for species in selected_species} # groups observations by species
    selected_observations_dict = {species: random.sample(observations_selected_species_dict[species], min_num_observations_per_species) for species in selected_species} # samples a fixed number of observations per species
    OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']
    points = np.concatenate([ee.FeatureCollection([ee.Feature(ee.Geometry.Point([observation[1], observation[2]])).set({'main_species': species, 'month': observation[0]}) for observation in selected_observations_dict[species]]).map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry())).getInfo()['features'] for species in selected_species]) # creates a tile centered on each observation
    points = [{**{key: value for key, value in point.items() if key != 'id'}} for point in points] # removes ID
    print(np.array(list(selected_observations_dict.values())).shape)
    print(f'{len(points)} points before removing overlaps')

    utils.save_geojson(features=points, path='species/points/species_points_overlapping.geojson') # saves all the points as a GeoJSON
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path='species/points/species_outer_tiles_overlapping.geojson')
    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    # keep species with at least some minimum number of observations
    species_with_minimum_count = []
    species_observations_counts = get_all_species_observation_counts(points, selected_species)

    for species in selected_species:
        if species_observations_counts[species] >= final_num_tiles_per_species: # if the species' number of observations is at least the desired value
            species_with_minimum_count.append(species)

    print(f'{len(species_with_minimum_count)} species with enough observations')

    points = [point for point in points if point['properties']['main_species'] in species_with_minimum_count] # keeps species with some minimum number of observations
    print(f'{len(points)} points after removing species with too few observations')

    # randomly select 100 species
    species_to_keep = random.sample(species_with_minimum_count, 100) # randomly keeps 100 species
    points = [point for point in points if point['properties']['main_species'] in species_to_keep]
    print(f'{len(points)} points after selecting {len(species_to_keep)} random species')

    points = list(itertools.chain(*[random.sample([point for point in points if point['properties']['main_species'] == species], final_num_tiles_per_species) for species in species_to_keep])) # randomly selects a fixed number of points per species
    print(f'{len(points)} points after ensuring each species has {final_num_tiles_per_species} observations')

    # record all species occurring in the points
    observations_selected_species_df = observations_selected_species_df[observations_selected_species_df['species'].isin(species_to_keep)] # only includes selected species
    observations_selected_species_df['geometry'] = observations_selected_species_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1) # adds a geometry column
    observations_selected_species_gdf = gpd.GeoDataFrame(observations_selected_species_df, geometry='geometry') # creates a GeoDataFrame for all the observations for the selected species
    observations_selected_species_gdf.to_file('species/observations_selected_species_gdf.geojson', driver='GeoJSON')

    # for point in points:
    #     point['properties']['main_species'] = point['properties']['species'] # the tile is centered on the main species' observation

        # month = point['properties']['month']

        # if month > 1 and month < 12:
        #     allowed_months = [month-1, month, month+1]
        # elif month == 1:
        #     allowed_months = [12, month, 2]
        # elif month == 12:
        #     allowed_months = [11, month, 1]

        # observations_in_tile = observations_selected_species_gdf[observations_selected_species_gdf.within(tile['properties']['outer_tile'])] # gets observations in the tile
        # observations_in_tile_filtered = observations_in_tile[observations_in_tile['month'].isin(allowed_months)] # keeps observations within one month of the main species'
        # tile['properties']['species'] = list(observations_in_tile_filtered['species'].unique()) # saves all relevant species to the tile

    # # plot species observation counts
    # fig, ax = plt.subplots(dpi=300)
    # ax.bar(species_to_keep, get_all_species_observation_counts(tiles, species_to_keep).values())
    # ax.set_ylabel('Observation count')
    # plt.xticks(rotation=90, fontsize=4)
    # plt.savefig('species/figures/species_observation_counts.pdf', bbox_inches='tight', pad_inches=0.1, transparent=False)
    # plt.savefig('species/figures/species_observation_counts.png', bbox_inches='tight', pad_inches=0.1, transparent=False)

    random.shuffle(points) # shuffles the points list
    points = [{**point, 'id': i} for i, point in enumerate(points)] # assigns each point an ID
    print(f'{len(points)} points, {len(species_to_keep)} species')

    utils.save_geojson(features=points, path='species/points/species_points.geojson') # saves all the points as a GeoJSON
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path='species/points/species_outer_tiles.geojson')

    # save list of species as a CSV
    with open('species/species.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for species in species_to_keep:
            writer.writerow([species])

    with open('species/species_labels.json', 'w') as file:
        json.dump({species: i for i, species in enumerate(species_to_keep)}, file, indent=4)

    # utils.make_global_map(tiles=tiles, color='r', path='species/figures/species_map', title='Species')

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def get_wikipedia_url(species):
    print('Wikipedia image')
    wikipedia_page_url = f'https://en.wikipedia.org/wiki/{species.replace(' ', '_')}' # URL for the Wikipedia page with the given title
    response = requests.get(wikipedia_page_url, headers={'User-Agent': 'LuciaGordon (https://lgordon99.github.io; luciagordon@g.harvard.edu)'})
    soup = BeautifulSoup(response.content, 'html.parser')
    url = f'https:{soup.find("table", {"class": "infobox"}).find("img")["src"]}'

    return url

def get_url(species):
    print(species)
    species_metadata = utils.read_json(f'{data_dir_path}/sinr-data/train/geo_prior_train_meta.json')
    metadata = next(item for item in species_metadata if item['latin_name'] == species)

    if 'amazon' in metadata['default_photo']:
        url = metadata['default_photo']

        try:
            subprocess.run(['wget', url, '-O', f'species/figures/species_images/{species}.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            Image.open(f'species/figures/species_images/{species}.jpg')
        except:
            url = get_wikipedia_url(species)
    else:
        url = get_wikipedia_url(species)

    print(url)

    return url

def make_species_grid():
    os.makedirs('species/figures/species_images', exist_ok=True)

    species_list = pd.read_csv('species/species.csv', header=None).values.tolist()

    for species in species_list:
        species = species[0]
        url = get_url(species)
        subprocess.run(['wget', url, '-O', f'species/figures/species_images/{species}.jpg'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axes.flat):
        if i < len(species_list):
            image = Image.open(f'species/figures/species_images/{species_list[i][0]}.jpg')
            ax.imshow(image)

        ax.axis('off')
        ax.set_title(species_list[i][0], fontsize=8)

    plt.tight_layout()
    plt.savefig('species/figures/species_grid.pdf', bbox_inches='tight')
    plt.savefig('species/figures/species_grid.png', bbox_inches='tight')

if __name__ == '__main__':
    if not os.path.exists(f'{data_dir_path}/sinr-data/observations_2020.csv'):
        save_2020_observations() # takes ~3 minutes

    if len(argv) == 1:
        generate_species_points() # takes ~22 minutes
    if len(argv) > 1:
        if argv[1] == 'make_species_grid':
            make_species_grid()
