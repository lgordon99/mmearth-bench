# ============================================== IMPORTS ============================================== #

from convert_to_h5 import get_tile_id
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from rasterio.warp import transform_bounds
from shapely.geometry import box
from sys import argv
from tqdm import tqdm
import geopandas as gpd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import subprocess
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')
# pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'DynamicWorld', 'ESA_WorldCover', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL']
pixel_level_modalities = ['Sentinel2', 'Sentinel1', 'ETH_GCH', 'DynamicWorld', 'ESA_WorldCover', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL']
biome_labels = utils.read_json('biomes_ecoregions_data/biome_labels.json')
ecoregion_labels = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')
precipitation_keys = ['Precipitation previous month', 'Precipitation this month', 'Precipitation year']
temperature_keys = ['Temperature previous month max',
                    'Temperature previous month mean',
                    'Temperature previous month min',
                    'Temperature this month max',
                    'Temperature this month mean',
                    'Temperature this month min',
                    'Temperature year max',
                    'Temperature year mean',
                    'Temperature year min']
dynamic_world_colors_labels = {'#419BDF': 'Water',
                               '#397D49': 'Trees',
                               '#88B053': 'Grass',
                               '#7A87C6': 'Flooded vegetation',
                               '#E49635': 'Crops',
                               '#DFC35A': 'Shrub and scrub',
                               '#C4281B': 'Built',
                               '#A59B8F': 'Bare',
                               '#B39FE1': 'Snow and ice'}
esa_worldcover_colors_labels = {'#006400': 'Tree cover',
                                '#ffbb22': 'Shrubland',
                                '#ffff4c': 'Grassland',
                                '#f096ff': 'Cropland',
                                '#fa0000': 'Built-up',
                                '#b4b4b4': 'Bare/sparse\nvegetation',
                                '#f0f0f0': 'Snow and\nice',
                                '#0064c8': 'Permanent\nwater bodies',
                                '#0096a0': 'Herbaceous\nwetland',
                                '#00cf75': 'Mangroves',
                                '#fae6a0': 'Moss and\nlichen'}
scl_colors_labels = {'#868686': 'Dark area',
                     '#774b0a': 'Cloud shadows',
                     '#10d22c': 'Vegetation',
                     '#ffff52': 'Bare soils',
                     '#0000ff': 'Water',
                     '#818181': 'Clouds low\nprobability\n/unclassified',
                     '#c0c0c0': 'Clouds\nmedium\nprobability',
                     '#f1f1f1': 'Clouds high\nprobability',
                     '#bac5eb': 'Cirrus',
                     '#52fff9': 'Snow/ice'}

# ============================================== FUNCTIONS ============================================== #

def normalize(array):
    for i in range(array.shape[0]):
        band = array[i]
        band_min = band.min()
        band_max = band.max()

        if np.ma.is_masked(band_min) or band_min == band_max:
            array[i] = 0
        else:
            array[i] = (band - band_min) / (band_max - band_min)

    return array

def create_categorical_legend(colors_labels_dict, output_path):
    if os.path.exists(output_path):
        return

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    n_items = len(colors_labels_dict)
    patch_width = 0.8 / n_items
    patch_height = 0.4
    patch_y = 0.5
    label_y_offset = 0.15

    for i, (color, label) in enumerate(colors_labels_dict.items()):
        x_center = (i + 0.5) / n_items
        x_left = x_center - patch_width / 2

        # Draw patch
        rect = plt.Rectangle((x_left, patch_y), patch_width, patch_height, facecolor=color, transform=ax.transAxes)
        ax.add_patch(rect)

        # Add label underneath
        ax.text(x_center,
                patch_y - label_y_offset,
                label,
                ha='center',
                va='top',
                fontsize=8,
                transform=ax.transAxes,
                rotation=0,
                wrap=True)

    plt.savefig(output_path, transparent=True, bbox_inches='tight', dpi=300)
    plt.close(fig)

def process_tile(tiff_name):
    tile_id = tiff_name.split('_')[1].split('.')[0]

    with rasterio.open(f'{data_dir_path}/{task}/tiffs/{tiff_name}') as tiff:
        crs = tiff.crs
        bounds = tiff.bounds
        tags = tiff.tags()
        band_names = {band_number: tiff.tags(band_number+1)['BAND_NAME'] for band_number in range(tiff.count)}
        array = tiff.read()
        pixel_level_data = {}

        # pixel-level data
        for modality in pixel_level_modalities:
            modality_band_numbers = [band_number for band_number, band_name in band_names.items() if modality in band_name] # extracts the band numbers for the modality
            pixel_level_data[modality] = np.ma.masked_equal(array[modality_band_numbers], no_data_values[modality]) # saves the modality data

        pixel_level_data['Sentinel2'] = np.stack(normalize(pixel_level_data['Sentinel2'][[3,2,1]].astype(float)).filled(0), axis=-1) # (H, W, 3)
        sentinel1 = np.zeros((pixel_level_data['Sentinel1'].shape[1:]))

        for band in pixel_level_data['Sentinel1']:
            if (~band.mask).sum() > 0: # if there are no masked pixels
                sentinel1 = band.astype(float)
                break

        pixel_level_data['Sentinel1'] = sentinel1
        # pixel_level_data['AsterDEM'] = pixel_level_data['AsterDEM'][0] # extracts the elevation data from the AsterDEM data
        pixel_level_data['ETH_GCH'] = pixel_level_data['ETH_GCH'][0] # extracts the height data from the ETH_GCH data

        if task == 'biomass':
            biomass = np.ma.masked_equal(array[[band_number for band_number, band_name in band_names.items() if band_name == 'biomass'][0]], -9999)

        # tile-level data
        if task != 'biomass':
            task_value = tags[task]

        sentinel2_date = tags['sentinel2_date']
        geolocation = json.loads(tags['geolocation'])
        latitude = round(geolocation[1], 4)
        longitude = round(geolocation[0], 4)
        biome = next((name for name, index in biome_labels.items() if str(index) == tags['biome']), None)
        ecoregion = next((name for name, index in ecoregion_labels.items() if str(index) == tags['ecoregion']), None)
        precipitation = {key: round(value, 2) for key, value in dict(zip(precipitation_keys, json.loads(tags['precipitation']))).items()}
        temperature = {key: round(value, 2) for key, value in dict(zip(temperature_keys, json.loads(tags['temperature']))).items()}
        msk_cldprb_cloudy_pixel_percentage = round(100 * float(tags['MSK_CLDPRB_CLOUDY_PIXEL_FRACTION']), 2)
        s2cloudless_cloudy_pixel_percentage = round(100 * float(tags['S2CLOUDLESS_CLOUDY_PIXEL_FRACTION']), 2)
        scl_no_data_pixel_percentage = round(100 * float(tags['SCL_NO_DATA_PIXEL_FRACTION']), 2)

    if crs != 'EPSG:4326':
        bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    tile_level_data = {'ID': tile_id,
                       'geometry': bbox,
                       'sentinel2_date': sentinel2_date,
                       'latitude': latitude,
                       'longitude': longitude,
                       'biome': biome,
                       'ecoregion': ecoregion,
                       **precipitation,
                       **temperature,
                       'MSK_CLDPRB_CLOUDY_PIXEL_PERCENTAGE': msk_cldprb_cloudy_pixel_percentage,
                       'S2CLOUDLESS_CLOUDY_PIXEL_PERCENTAGE': s2cloudless_cloudy_pixel_percentage,
                       'SCL_NO_DATA_PIXEL_PERCENTAGE': scl_no_data_pixel_percentage}

    if task != 'biomass':
        tile_level_data[task] = task_value

    for modality in pixel_level_modalities:
        os.makedirs(f'{data_dir_path}/{task}/png_tiles/{modality}', exist_ok=True)

        if modality == 'ETH_GCH':
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='black')
            norm = mpl.colors.Normalize(vmin=0, vmax=70)
            pixel_level_data[modality] = cmap(norm(pixel_level_data[modality]))

            # save colorbar
            if not os.path.exists(f'{data_dir_path}/map_legend_{modality.lower()}.png'):
                fig, ax = plt.subplots(figsize=(6, 0.5))
                cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
                cbar.set_label('Canopy height (m)', labelpad=10, fontsize=8)
                cbar.ax.tick_params(labelsize=8)
                plt.savefig(f'{data_dir_path}/map_legend_{modality.lower()}.png', transparent=True, bbox_inches='tight', dpi=300)
                plt.close(fig)
        elif modality == 'DynamicWorld':
            cmap = ListedColormap(dynamic_world_colors_labels.keys())
            cmap.set_bad(color='black')
            pixel_level_data[modality] = cmap(pixel_level_data[modality].astype(int))
            create_categorical_legend(dynamic_world_colors_labels, f'{data_dir_path}/map_legend_dynamicworld.png')
        elif modality == 'ESA_WorldCover':
            cmap = ListedColormap(esa_worldcover_colors_labels.keys())
            cmap.set_bad(color='black')
            pixel_level_data[modality] = cmap(pixel_level_data[modality].astype(int))
            create_categorical_legend(esa_worldcover_colors_labels, f'{data_dir_path}/map_legend_esa_worldcover.png')
        elif modality == 'MSK_CLDPRB' or modality == 'S2CLOUDLESS':
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='black')
            norm = mpl.colors.Normalize(vmin=0, vmax=100)
            pixel_level_data[modality] = cmap(norm(pixel_level_data[modality]))

            # save colorbar
            if not os.path.exists(f'{data_dir_path}/map_legend_{modality.lower()}.png'):
                fig, ax = plt.subplots(figsize=(6, 0.5))
                cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
                cbar.set_label('Cloudy pixel probability (%)', labelpad=10, fontsize=8)
                cbar.ax.tick_params(labelsize=8)
                plt.savefig(f'{data_dir_path}/map_legend_{modality.lower()}.png', transparent=True, bbox_inches='tight', dpi=300)
                plt.close(fig)
        elif modality == 'SCL':
            cmap = ListedColormap(scl_colors_labels.keys())
            cmap.set_bad(color='black')
            pixel_level_data[modality] = cmap(pixel_level_data[modality].astype(int) - 2) # subtract 2 to start at 0
            create_categorical_legend(scl_colors_labels, f'{data_dir_path}/map_legend_scl.png')

        plt.imsave(f'{data_dir_path}/{task}/png_tiles/{modality}/tile_{tile_id}_{modality}.png', pixel_level_data[modality].squeeze())

    if task == 'biomass':
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='black')
        norm = mpl.colors.Normalize(vmin=0, vmax=2000)
        os.makedirs(f'{data_dir_path}/{task}/png_tiles/biomass', exist_ok=True)
        plt.imsave(f'{data_dir_path}/{task}/png_tiles/biomass/tile_{tile_id}_biomass.png', cmap(norm(biomass)))

        # save colorbar
        if not os.path.exists(f'{data_dir_path}/{task}/map_legend_biomass.png'):
            fig, ax = plt.subplots(figsize=(6, 0.5))
            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
            cbar.set_label('Biomass value (Mg/ha)', labelpad=10, fontsize=8)
            cbar.ax.tick_params(labelsize=8)
            plt.savefig(f'{data_dir_path}/{task}/map_legend_biomass.png', transparent=True, bbox_inches='tight', dpi=300)
            plt.close(fig)

    return tile_level_data

def save_map_data(task):
    tiffs = sorted(os.listdir(f'{data_dir_path}/{task}/tiffs'), key=get_tile_id)

    with Pool() as pool: # parallel processing
        tile_level_data_list = list(tqdm(pool.imap(process_tile, tiffs), total=len(tiffs)))

    gpd.GeoDataFrame(tile_level_data_list, crs='EPSG:4326').to_file(f'{data_dir_path}/{task}/{task}_map_gdf.geojson', driver='GeoJSON')

if __name__ == '__main__':
    if 'for' not in argv[1]: # python generate_map_data.py TASK
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        subprocess.run(['sbatch', '-t', '0-5:00:00', '-p', partitions, '--mem', '500M', '--job-name', f'{argv[1]}_map_data', '-o', f'bash-outputs/{argv[1]}_map_data.out', '-e', f'bash-errors/{argv[1]}_map_data.err', 'job.sh', env_path, 'generate_map_data.py', f'for_{argv[1]}'])
    else: # python generate_map_data.py for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        save_map_data(task)
