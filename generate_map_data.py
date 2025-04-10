# ============================================== IMPORTS ============================================== #

from rasterio.warp import transform_bounds
from shapely.geometry import box
from sys import argv
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import rasterio
import subprocess
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
month_labels = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
biome_labels = utils.read_json('biomes_ecoregions_data/biome_labels.json')
ecoregion_labels = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')

# ============================================== FUNCTIONS ============================================== #

def save_map_data(task):
    gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

    os.makedirs(f'{data_dir_path}/{task}/tiles', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/Sentinel-2', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/Sentinel-1', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/AsterDEM-elevation', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/ETHGCH-canopy-height', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/DynamicWorld', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/ESA-Worldcover', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/MSK_CLDPRB', exist_ok=True)
    os.makedirs(f'{data_dir_path}/{task}/tiles/S2CLOUDLESS', exist_ok=True)

    if task == 'biomass':
        os.makedirs(f'{data_dir_path}/{task}/tiles/biomass', exist_ok=True)

    for tiff_name in os.listdir(f'{data_dir_path}/{task}/data'):
        tile_id = tiff_name.split('_')[1]

        with rasterio.open(f'{data_dir_path}/{task}/data/{tiff_name}') as tiff:
            crs = tiff.crs
            bounds = tiff.bounds
            tags = tiff.tags()
            band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)}
            array = tiff.read()

            rgb = array[[3,2,1]].astype(float) # R, G, B
            s1 = np.zeros((array.shape[1:]))
            sentinel1_band_numbers = [band_number for band_number, band_name in band_names.items() if 'Sentinel1' in band_name]

            for band_number in sentinel1_band_numbers:
                if np.all(array[band_number] != -9999):
                    s1 = array[band_number].astype(float)
                    break

            # pixel-level data
            asterdem_elevation = array[[band_number for band_number, band_name in band_names.items() if 'AsterDEM_elevation' in band_name][0]]
            ethgch_canopy_height = array[[band_number for band_number, band_name in band_names.items() if 'ETHGCH_canopy_height' in band_name][0]]
            dynamicworld = array[[band_number for band_number, band_name in band_names.items() if 'DynamicWorld' in band_name][0]]
            esa_worldcover = array[[band_number for band_number, band_name in band_names.items() if 'ESA_Worldcover' in band_name][0]]
            msk_cldprb = array[[band_number for band_number, band_name in band_names.items() if 'MSK_CLDPRB' in band_name][0]]
            s2cloudless = array[[band_number for band_number, band_name in band_names.items() if 'S2CLOUDLESS' in band_name][0]]

            if task == 'biomass':
                biomass = array[[band_number for band_number, band_name in band_names.items() if 'biomass' in band_name][0]]

            # image-level data
            if task != 'biomass':
                task_value = tags[task] if task != 'species' else tags['name_species']

            climate = {key.split('climate_')[1].capitalize().replace('_', ' '): value for key, value in tags.items() if 'climate' in key}
            latitude = tags['lat']
            longitude = tags['lon']
            month = next((key for key, value in month_labels.items() if value == int(tags['s2_date'].split('-')[1])), None)
            biome = next((key for key, value in biome_labels.items() if str(value) == tags['biome']), None)
            ecoregion = next((key for key, value in ecoregion_labels.items() if str(value) == tags['ecoregion']), None)
            msk_cldprb_cloudy_pixel_fraction = tags['MSK_CLDPRB_CLOUDY_PIXEL_FRACTION']
            s2cloudless_cloudy_pixel_fraction = tags['S2CLOUDLESS_CLOUDY_PIXEL_FRACTION']

        if crs != 'EPSG:4326':
            bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        image_level_data = {'id': tile_id,
                            'geometry': bbox,
                            'climate': climate,
                            'latitude': latitude,
                            'longitude': longitude,
                            'month': month,
                            'biome': biome,
                            'ecoregion': ecoregion,
                            'msk_cldprb_cloudy_pixel_fraction': msk_cldprb_cloudy_pixel_fraction,
                            's2cloudless_cloudy_pixel_fraction': s2cloudless_cloudy_pixel_fraction}

        if task != 'biomass':
            image_level_data[task] = task_value

        gdf = pd.concat([gdf, gpd.GeoDataFrame([image_level_data], crs=gdf.crs)], ignore_index=True)
        gdf.to_file(f'{task}/{task}_tile_gdf.geojson', driver='GeoJSON')

        rgb = np.stack(utils.normalize(rgb), axis=-1) # (H, W, 3)
        msk_cldprb = np.nan_to_num(msk_cldprb, nan=100)
        plt.imsave(f'{data_dir_path}/{task}/tiles/Sentinel-2/tile_{tile_id}_Sentinel-2.png', rgb)
        plt.imsave(f'{data_dir_path}/{task}/tiles/Sentinel-1/tile_{tile_id}_Sentinel-1.png', s1)
        plt.imsave(f'{data_dir_path}/{task}/tiles/AsterDEM-elevation/tile_{tile_id}_AsterDEM-elevation.png', asterdem_elevation)
        plt.imsave(f'{data_dir_path}/{task}/tiles/ETHGCH-canopy-height/tile_{tile_id}_ETHGCH-canopy-height.png', ethgch_canopy_height)
        plt.imsave(f'{data_dir_path}/{task}/tiles/DynamicWorld/tile_{tile_id}_DynamicWorld.png', dynamicworld)
        plt.imsave(f'{data_dir_path}/{task}/tiles/ESA-Worldcover/tile_{tile_id}_ESA-Worldcover.png', esa_worldcover)
        plt.imsave(f'{data_dir_path}/{task}/tiles/MSK_CLDPRB/tile_{tile_id}_MSK_CLDPRB.png', msk_cldprb)
        plt.imsave(f'{data_dir_path}/{task}/tiles/S2CLOUDLESS/tile_{tile_id}_S2CLOUDLESS.png', s2cloudless)

        if task == 'biomass':
            biomass = ma.masked_equal(biomass, -9999) # masks nodata values
            cmap = plt.get_cmap('gnuplot2')
            cmap.set_bad(color='black')
            plt.imsave(f'{data_dir_path}/{task}/tiles/biomass/tile_{tile_id}_biomass.png', biomass, cmap=cmap)

if __name__ == '__main__':
    if 'for' not in argv[1]: # python generate_map_data.py TASK
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        subprocess.run(['sbatch', '-t', '0-5:00:00', '-p', partitions, '--mem', '500M', '--job-name', f'{argv[1]}_map_data', '-o', f'bash-outputs/{argv[1]}_map_data.out', '-e', f'bash-errors/{argv[1]}_map_data.err', 'job.sh', env_path, 'generate_map_data.py', f'for_{argv[1]}'])
    else: # python generate_map_data.py for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        save_map_data(task)
