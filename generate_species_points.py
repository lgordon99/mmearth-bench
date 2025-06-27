# REQUIRES 50GB RAM

# ============================================== IMPORTS ============================================== #

from shapely.geometry import Point, shape
from shapely.ops import unary_union
from tqdm import tqdm
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

ee.Initialize(project='mmearth-bench') # initializes EE with our project
random.seed(42)  # sets a seed for reproducibility
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
africa_gdf = gpd.read_file(f'{data_dir_path}/africa.geojson')
africa_union = unary_union(africa_gdf.geometry)

def get_species_ranges():
    mammals_gdf = gpd.read_file(f'{data_dir_path}/species/MAMMALS_TERRESTRIAL_ONLY/MAMMALS_TERRESTRIAL_ONLY.shp').dissolve(by='sci_name')
    print(mammals_gdf)
    mammals_in_africa = gpd.sjoin(mammals_gdf, africa_gdf) # 1442 species
    print(mammals_in_africa)
    mammals_in_africa['area_in_africa'] = mammals_in_africa.geometry.intersection(africa_union).area
    mammals_in_africa['area_outside_africa'] = mammals_in_africa.geometry.difference(africa_union).area
    print(mammals_in_africa)
    mammals_in_africa = mammals_in_africa[mammals_in_africa['area_in_africa'] >= 1] # 1091 species
    print(mammals_in_africa)
    mammals_in_out_africa = mammals_in_africa[mammals_in_africa['area_outside_africa'] >= 1] # 98 species
    print(mammals_in_out_africa)
    os.makedirs(f'{data_dir_path}/species/mammals_in_out_africa', exist_ok=True)
    mammals_in_out_africa.to_file(f'{data_dir_path}/species/mammals_in_out_africa/mammals_in_out_africa.shp', driver='ESRI Shapefile')

def generate_species_points():
    start_time = time.time()
    species_gdf = gpd.read_file(f'{data_dir_path}/species/mammals_in_out_africa/mammals_in_out_africa.shp')
    OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']
    os.makedirs('species/points', exist_ok=True)
    points = []

    for row in tqdm(species_gdf.itertuples(index=False), total=len(species_gdf), desc='Species'):
        species = row.sci_name
        print(species)

        range_in_africa = row.geometry.intersection(africa_union)
        range_outside_africa = row.geometry.difference(africa_union)
        split_data = {'in_africa': {'num_tiles': 100, 'range': range_in_africa, 'points': []},
                      'outside_africa': {'num_tiles': 300, 'range': range_outside_africa, 'points': []}}

        for split, data in split_data.items():
            while len(split_data[split]['points']) < data['num_tiles']:
                minx, miny, maxx, maxy = data['range'].bounds
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                point = Point(x, y)

                if data['range'].contains(point):
                    split_data[split]['points'].append(ee.Feature(ee.Geometry.Point([x, y])).set({'species': [species], 'outer_tile': ee.Geometry.Point([x, y]).buffer(OUTER_TILE_SIZE_M / 2).bounds()}).getInfo()) # creates an outer tile around each point

        species_points = [point for split in split_data.keys() for point in split_data[split]['points']]
        print(f'{len(species_points)} before removing overlaps')

        species_points = utils.remove_overlapping_tiles(species_points) # removes points with overlapping tiles
        print(f'{len(species_points)} points after removing overlaps')

        points.extend(species_points)

    print(f'{len(points)} before removing overlaps')
    utils.save_geojson(features=points, path='species/points/species_points_overlapping.geojson') # saves all the points as a GeoJSON
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path='species/points/species_outer_tiles_overlapping.geojson')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    for point in points:
        for row in species_gdf.itertuples(index=False):
            if row.geometry.contains(shape(point['properties']['outer_tile'])): # if the outer tile of the point is within the species range
                point['properties']['species'].append(row.sci_name)

    random.shuffle(points) # shuffles the points list
    points = [{**point, 'id': i} for i, point in enumerate(points)] # assigns each point an ID
    utils.save_geojson(features=points, path='species/points/species_points.geojson') # saves the points as a GeoJSON
    utils.save_geojson(features=[{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': point['properties']['outer_tile']['coordinates']}} for point in points], path='species/points/species_outer_tiles.geojson')

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_point_statistics():
    os.makedirs('species/figures', exist_ok=True)
    points = utils.read_geojson('species/points/species_points.geojson')['features']
    print(f'Total points: {len(points)}')

    # plot of number of tiles per species
    species_counts = {}

    for point in points:
        for species in point['properties']['species']:
            if species in species_counts.keys():
                species_counts[species] += 1
            else:
                species_counts[species] = 1

    species = list(species_counts.keys())
    counts = list(species_counts.values())
    indices = np.arange(len(species))
    plt.figure(dpi=300, figsize=(15, 5))
    plt.bar(indices, counts)
    plt.xticks(indices, species, rotation=90)
    plt.xlabel('Species')
    plt.ylabel('Number of tiles')
    plt.title('Number of tiles per species')
    plt.tight_layout()
    plt.savefig('species/figures/species_counts.png')

    print(f'Max number of tiles for a species: {max(counts)}')
    print(f'Min number of tiles for a species: {min(counts)}')

    # histogram of number of species per tile
    tile_species_counts = []

    for point in points:
        tile_species_counts.append(len(point['properties']['species']))

    plt.figure(dpi=300, figsize=(15, 5))
    plt.hist(tile_species_counts, bins=np.arange(min(tile_species_counts), max(tile_species_counts) + 2) - 0.5, edgecolor='black')
    plt.xlabel('Number of species')
    plt.ylabel('Number of tiles')
    plt.title('Number of species per tile')
    plt.tight_layout()
    plt.savefig('species/figures/tile_species_counts.png')

if not os.path.exists(f'{data_dir_path}/species/mammals_in_out_africa/mammals_in_out_africa.shp'):
    get_species_ranges()

if not os.path.exists('species/points/species_points.geojson'):
    generate_species_points()

check_point_statistics()
