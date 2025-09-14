# REQUIRES 10GB RAM

# ============================================== IMPORTS ============================================== #

from shapely.geometry import shape
from sys import argv
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import subprocess
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

random.seed(42) # sets a seed for reproducibility
ee.Initialize(project='mmearth-bench') # initializes EE with the project
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment

def get_species_ranges():
    with open(f'{data_dir_path}/species/output-files/get_species_ranges.out', 'w') as out_file:
        mammals_gdf = gpd.read_file(f'{data_dir_path}/species/MAMMALS_TERRESTRIAL_ONLY/MAMMALS_TERRESTRIAL_ONLY.shp').dissolve(by='sci_name').reset_index() # 5,675 species, combines multiple Polygons for each species into one MultiPolygon
        mammals_gdf = mammals_gdf[['sci_name', 'geometry', 'order_',]]
        out_file.write(f'{len(mammals_gdf)} total species\n')
        africa_gdf = gpd.read_file(f'{data_dir_path}/africa.geojson') # Africa boundaries
        mammals_in_africa = mammals_gdf[mammals_gdf.intersects(africa_gdf.geometry.squeeze())] # 1,451 species in Africa
        out_file.write(f'{len(mammals_in_africa)} species in Africa\n')
        global_equal_area_crs = 'EPSG:6933' # global equal area projection
        reprojected_mammals = mammals_in_africa.to_crs(global_equal_area_crs).copy() # reprojects species ranges to equal area projection
        reprojected_mammals['geometry'] = reprojected_mammals.geometry.make_valid() # fixes invalid geometries
        reprojected_africa_geometry = africa_gdf.to_crs(global_equal_area_crs).geometry.squeeze() # reprojects Africa boundaries to equal area projection
        area_in_africa = reprojected_mammals.geometry.intersection(reprojected_africa_geometry).area / 1e6 # area of each species range within Africa in km^2
        area_outside_africa = reprojected_mammals.geometry.difference(reprojected_africa_geometry).area / 1e6 # area of each species range outside Africa in km^2
        mammals_in_africa = mammals_in_africa.assign(area_in_africa=area_in_africa.values, area_outside_africa=area_outside_africa.values) # adds area columns to the GeoDataFrame
        mammals_in_africa = mammals_in_africa[mammals_in_africa['area_outside_africa'] > 0] # 122 species
        out_file.write(f'{len(mammals_in_africa)} species found in and outside Africa\n')
        mammals_in_africa = mammals_in_africa[(mammals_in_africa['area_outside_africa'] >= 6000) & (mammals_in_africa['area_in_africa'] >= 6000)]
        out_file.write(f'{len(mammals_in_africa)} species whose range covers at least 6000 km^2 both in and outside Africa\n')
        mammals_in_africa = mammals_in_africa[:100] # limits to 100 species
        out_file.write(f'{len(mammals_in_africa)} species selected\n')
        out_file.write(f'Min area in Africa: {round(mammals_in_africa["area_in_africa"].min(), 2)} km^2, min area outside Africa: {round(mammals_in_africa["area_outside_africa"].min(), 2)} km^2\n')
        mammals_in_africa.to_file(f'{data_dir_path}/species/species_ranges.geojson', driver='GeoJSON')

    # plot species per order
    order_counts = mammals_in_africa['order_'].value_counts()
    order_counts.index = order_counts.index.str.capitalize()
    plt.figure(dpi=300)
    order_counts.plot(kind='bar')
    plt.xlabel('Order')
    plt.ylabel('Number of species')
    plt.title('Number of species per order')
    plt.tight_layout()
    plt.xticks(rotation=15)
    plt.savefig(f'{data_dir_path}/species/species_per_order.png')

def generate_species_points():
    start_time = time.time()
    species_gdf = gpd.read_file(f'{data_dir_path}/species/species_ranges.geojson')
    OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']
    africa_geometry = gpd.read_file(f'{data_dir_path}/africa.geojson').geometry.squeeze()
    points = []

    for i, row in enumerate(species_gdf.itertuples(index=False)):
        species = row.sci_name
        print(f'Species {i+1}/{len(species_gdf)}: {species}')
        range_in_africa = row.geometry.intersection(africa_geometry)
        range_outside_africa = row.geometry.difference(africa_geometry)
        split_data = {'in_africa': {'num_tiles': 100, 'range': range_in_africa, 'points': []},
                      'outside_africa': {'num_tiles': 300, 'range': range_outside_africa, 'points': []}}

        for split, data in split_data.items():
            minx, miny, maxx, maxy = data['range'].bounds # bounding box of the range

            while len(split_data[split]['points']) < data['num_tiles']:
                x = random.uniform(minx, maxx) # random x coordinate within the bounding box
                y = random.uniform(miny, maxy) # random y coordinate within the bounding box
                point = ee.Feature(ee.Geometry.Point([x, y])).set({'species': [species], 'outer_tile': ee.Geometry.Point([x, y]).buffer(OUTER_TILE_SIZE_M / 2).bounds()}).getInfo()

                if data['range'].intersects(shape(point['properties']['outer_tile'])): # if the point's outer tile intersects the range
                    split_data[split]['points'].append(point)

        species_points = [point for split in split_data.keys() for point in split_data[split]['points']]
        print(f'{len(species_points)} points before removing overlaps')
        species_points = utils.remove_overlapping_tiles(species_points) # removes points with overlapping tiles
        print(f'{len(species_points)} points after removing overlaps')
        points.extend(species_points)

    print(f'{len(points)} points before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    for point in points:
        for row in species_gdf.itertuples(index=False): # for each species
            if row.sci_name not in point['properties']['species'] and row.geometry.intersects(shape(point['properties']['outer_tile'])): # if the point's outer tile intersects the species range
                point['properties']['species'].append(row.sci_name) # adds the species to the point's species list

    random.shuffle(points) # shuffles the points list
    points = [{**{key: value for key, value in point.items() if key != 'properties'}, 'properties': {key: value for key, value in point['properties'].items() if key != 'outer_tile'}, 'id': i} for i, point in enumerate(points)] # assigns each point an ID and removes the outer tile
    utils.save_geojson(features=points, path=f'{data_dir_path}/species/species_points.geojson') # saves the points as a GeoJSON

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_point_statistics():
    with open(f'{data_dir_path}/species/output-files/check_point_statistics.out', 'w') as out_file:
        points = utils.read_geojson(f'{data_dir_path}/species/species_points.geojson')['features']
        out_file.write(f'Total points: {len(points)}\n')

        # plot number of points per species
        species_counts = {}

        for point in points:
            for species in point['properties']['species']:
                species_counts[species] = species_counts.get(species, 0) + 1

        species = list(species_counts.keys())
        counts = list(species_counts.values())
        indices = np.arange(len(species))
        plt.figure(dpi=300, figsize=(20, 5))
        plt.bar(indices, counts)
        plt.xticks(indices, species, rotation=90)
        plt.xlabel('Species')
        plt.ylabel('Number of points')
        plt.title('Number of points per species')
        plt.tight_layout()
        plt.savefig(f'{data_dir_path}/species/points_per_species.png')

        out_file.write(f'Max number of points for a species: {max(counts)}\n')
        out_file.write(f'Min number of points for a species: {min(counts)}\n')

        # histogram of number of species per point
        plt.figure(dpi=300)
        bin_size = 1000
        bins = np.arange(0, max(counts) + bin_size, bin_size)
        tick_interval = 5000

        plt.hist(counts, bins=bins, edgecolor='black')
        plt.xlabel('Number of points')
        plt.ylabel('Number of species')
        plt.xticks(np.arange(0, max(counts) + tick_interval, tick_interval))
        plt.tight_layout()
        plt.savefig(f'{data_dir_path}/species/point_species_counts.png')

if __name__ == '__main__':
    if len(argv) == 1:
        if not os.path.exists(f'{data_dir_path}/species/species_ranges.geojson'):
            get_species_ranges()

        if not os.path.exists(f'{data_dir_path}/species/species_points.geojson'):
            subprocess.run(['sbatch', '-t', '1-00:00', '-p', partitions, '--mem', '1G', '--job-name', 'generate_species_points', '-o', f'{data_dir_path}/species/output-files/generate_species_points.out', '--account', 'davies_lab', 'job.sh', env_path, 'generate_species_points.py', 'generate_species_points'])
    elif len(argv) > 1:
        if argv[1] == 'generate_species_points':
            generate_species_points()
        elif argv[1] == 'check_point_statistics':
            check_point_statistics()
