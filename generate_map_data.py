# ============================================== IMPORTS ============================================== #

from rasterio.warp import transform_bounds
from shapely.geometry import box
from sys import argv
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio
import utils

env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
task = argv[1]
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
data_dir = os.listdir(f'{data_dir_path}/{task}/data')
month_labels = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
biome_labels = utils.read_json('biomes_ecoregions_data/biome_labels.json')
ecoregion_labels = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')

def normalize(array):
    for i in range(array.shape[0]):
        if array[i].max() != array[i].min(): # check to avoid division by zero
            array[i] = (array[i] - array[i].min()) / (array[i].max() - array[i].min())
        else:
            array[i] = np.zeros_like(array[i]) # assigns a default value when all elements in the band are the same

    return array

def convert_tiffs_to_wmts_tiles(task):
    gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
    bounds_dict = {}
    os.makedirs(f'{data_dir_path}/{task}/tiles', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/Sentinel-2', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/Sentinel-1', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/AsterDEM-elevation', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/ETHGCH-canopy-height', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/DynamicWorld', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/ESA-Worldcover', exist_ok=True)

    for tiff_name in data_dir:
        name = tiff_name.split('data.tif')[0]

        with rasterio.open(f'{data_dir_path}/{task}/data/{tiff_name}') as tiff:
            crs = tiff.crs
            bounds = tiff.bounds
            tags = tiff.tags()
            band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)}
            sentinel1_band_numbers = [band_number for band_number, band_name in band_names.items() if 'Sentinel1' in band_name]
            array = tiff.read()

            rgb = array[[3,2,1]].astype(float) # R, G, B
            s1 = np.zeros((array.shape[1:]))

            # for i in range(1, tiff.count + 1):  # Bands are 1-indexed
            #     print(f"Band {i}: {tiff.tags(i)}")

            for band_number in sentinel1_band_numbers:
                if np.all(array[band_number] != -9999):
                    s1 = array[band_number].astype(float)
                    break

            asterdem_elevation = array[[band_number for band_number, band_name in band_names.items() if 'AsterDEM_elevation' in band_name][0]]
            ethgch_canopy_height = array[[band_number for band_number, band_name in band_names.items() if 'ETHGCH_canopy_height' in band_name][0]]
            dynamicworld = array[[band_number for band_number, band_name in band_names.items() if 'DynamicWorld' in band_name][0]]
            esa_worldcover = array[[band_number for band_number, band_name in band_names.items() if 'ESA_Worldcover' in band_name][0]]

            task_value = tags[task] if task != 'species' else tags['name_species']
            climate = {key.split('climate_')[1].capitalize().replace('_', ' '): value for key, value in tags.items() if 'climate' in key}
            latitude = tags['lat']
            longitude = tags['lon']
            month = next((key for key, value in month_labels.items() if value == int(tags['s2_date'].split('-')[1])), None)
            biome = next((key for key, value in biome_labels.items() if str(value) == tags['biome']), None)
            ecoregion = next((key for key, value in ecoregion_labels.items() if str(value) == tags['ecoregion']), None)

        if crs != 'EPSG:4326':
            bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        gdf = pd.concat([gdf, gpd.GeoDataFrame([{'geometry': bbox,
                                                 task: task_value,
                                                 'climate': climate,
                                                 'latitude': latitude,
                                                 'longitude': longitude,
                                                 'month': month,
                                                 'biome': biome,
                                                 'ecoregion': ecoregion}], crs=gdf.crs)], ignore_index=True)
        gdf.to_file(f'{task}/{task}_tile_gdf.geojson', driver='GeoJSON')

        rgb = np.stack(normalize(rgb), axis=-1) # (H, W, 3)
        plt.imsave(f'{data_dir_path}/{task}/tiles/Sentinel-2/{name}Sentinel-2.png', rgb)
        plt.imsave(f'{data_dir_path}/{task}/tiles/Sentinel-1/{name}Sentinel-1.png', s1)
        plt.imsave(f'{data_dir_path}/{task}/tiles/AsterDEM-elevation/{name}AsterDEM-elevation.png', asterdem_elevation)
        plt.imsave(f'{data_dir_path}/{task}/tiles/ETHGCH-canopy-height/{name}ETHGCH-canopy-height.png', ethgch_canopy_height)
        plt.imsave(f'{data_dir_path}/{task}/tiles/DynamicWorld/{name}DynamicWorld.png', dynamicworld)
        plt.imsave(f'{data_dir_path}/{task}/tiles/ESA-Worldcover/{name}ESA-Worldcover.png', esa_worldcover)

        bounds_dict[name] = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    with open(f'{task}/{task}_tile_bounds.json', 'w') as file:
        json.dump(bounds_dict, file, indent=4)

convert_tiffs_to_wmts_tiles(task)
