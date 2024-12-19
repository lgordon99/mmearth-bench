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
import subprocess
import utils

env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
task = argv[1]
data_dir = os.listdir(f'{task}/data')

def generate_gdf(task):
    gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

    for tiff_name in data_dir:
        with rasterio.open(f'{task}/data/{tiff_name}') as tiff:
            crs = tiff.crs
            bounds = tiff.bounds
            task_value = tiff.tags()[task]

        if crs != 'EPSG:4326':
            bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        gdf = pd.concat([gdf, gpd.GeoDataFrame([{'geometry': bbox, task: task_value}], crs=gdf.crs)], ignore_index=True)
        gdf.to_file(f'{task}/{task}_tile_gdf.geojson', driver='GeoJSON')

def convert_tiffs_to_wmts_tiles(task):
    bounds_dict = {}
    # os.makedirs(f'{task}/rgb', exist_ok=True)
    os.makedirs(f'{task}/tiles', exist_ok=True)
    os.makedirs(f'{task}/tiles/Sentinel-2', exist_ok=True)

    for tiff_name in data_dir:
        # tiff_path = f'{datadir}/{tiff_name}'
        name = tiff_name.split('data.tif')[0]

        with rasterio.open(f'{task}/data/{tiff_name}') as tiff:
        #     rgb_tiff_path = f'{task}/rgb/{tiff_name.split(".")[0]}_rgb.tif'
        #     subprocess.run([f'{env_path}/bin/gdal_translate', '-b', '4', '-b', '3', '-b', '2', '-ot', 'Byte', '-scale', tiff_path, rgb_tiff_path])
        #     subprocess.run([f'{env_path}/bin/gdal2tiles.py', '-z', '2-10', rgb_tiff_path, f'{task}/tiles/{rgb_tiff_path.split(".")[0]}'])

            crs = tiff.crs
            bounds = tiff.bounds
            rgb = tiff.read([4,3,2]).astype(float) # R, G, B

        if crs != 'EPSG:4326':
            bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

        for i in range(rgb.shape[0]):
            if rgb[i].max() != rgb[i].min(): # check to avoid division by zero
                rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())
            else:
                rgb[i] = np.zeros_like(rgb[i]) # assigns a default value when all elements in the band are the same

        rgb = np.stack(rgb, axis=-1) # (H, W, 3)
        plt.imsave(f'{task}/tiles/Sentinel-2/{name}Sentinel-2.png', rgb)
        bounds_dict[name] = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
        # bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        # gdf = pd.concat([gdf, gpd.GeoDataFrame([{'geometry': bbox, task: task_value}], crs=gdf.crs)], ignore_index=True)
        # gdf.to_file(f'{task}/{task}_tile_gdf.geojson', driver='GeoJSON')

    with open(f'{task}/{task}_Sentinel-2_tile_bounds.json', 'w') as file:
        json.dump(bounds_dict, file, indent=4)

generate_gdf(task)
convert_tiffs_to_wmts_tiles(task)
