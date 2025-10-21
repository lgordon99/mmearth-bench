# ============================================== IMPORTS ============================================== #

from affine import Affine
from itertools import chain
from matplotlib.lines import Line2D
from rasterio.warp import transform_bounds
from shapely.geometry import box
from sys import argv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

random.seed(42)
training_fraction = 0.7
validation_fraction = 0.15
height, width = 128, 128
splits = ['train_100%', 'val', 'random_test', 'geographic_test']
subset_splits = ['train_100%', 'train_50%', 'train_5%']
all_splits = list(set(splits + subset_splits))
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')

# ============================================== FUNCTIONS ============================================== #

def get_box_wgs_84(transform, width, height, crs):
    affine_transform = Affine(*transform.tolist())
    left, top = affine_transform * (0, 0)
    right, bottom = affine_transform * (width, height)
    bounds = (left, bottom, right, top)

    if crs != 'EPSG:4326':
        bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

    return box(*bounds)

def generate_splits(task):
    with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
        transforms = h5_file['transform'][:] # affine transformation for each tile
        crs = h5_file['crs'].asstr()[...] # coordinate reference system for each tile
        task_data = h5_file[task][:]
        tile_count = len(transforms)

        # bounding boxes
        boxes = {i: get_box_wgs_84(transform=transforms[i], width=width, height=height, crs=crs[i]) for i in range(tile_count)} # dictionary of boxes for each tile
        africa = gpd.read_file(f'{data_dir_path}/africa.geojson')
        africa_boxes = {tile_index: box for tile_index, box in boxes.items() if box.intersects(africa.geometry.squeeze())} # dictionary of boxes for each tile intersecting the Africa boundaries
        non_africa_boxes = {tile_index: box for tile_index, box in boxes.items() if tile_index not in africa_boxes.keys()} # dictionary of boxes for each tile outside the Africa boundaries

        # training, validation, and testing tile indices
        non_africa_tile_indices = sorted(list(map(int, non_africa_boxes.keys()))) # indices for tiles outside Africa
        random.shuffle(non_africa_tile_indices) # randomly reorders the non-Africa-tile indices
        end_train_indices = int(training_fraction * len(non_africa_tile_indices)) # 70% of the non-Africa tiles for training
        end_val_indices = int((training_fraction+validation_fraction) * len(non_africa_tile_indices)) # 15% of the non-Africa tiles for validation
        split_data = {}
        split_data['train_100%_indices'] = sorted(non_africa_tile_indices[:end_train_indices])
        split_data['train_50%_indices'] = random.sample(split_data['train_100%_indices'], round(0.5 * len(split_data['train_100%_indices'])))
        split_data['train_5%_indices'] = random.sample(split_data['train_50%_indices'], round(0.1 * len(split_data['train_50%_indices'])))
        split_data['val_indices'] = sorted(non_africa_tile_indices[end_train_indices:end_val_indices])
        split_data['random_test_indices'] = sorted(non_africa_tile_indices[end_val_indices:]) # remaining 15% of the non-Africa tiles for testing
        split_data['geographic_test_indices'] = sorted(list(map(int, africa_boxes.keys()))) # Africa tiles for testing

        if task == 'species':
            species = [json.loads(lst) for lst in h5_file['species'].asstr()[...]]
            species_names = np.unique(list(chain.from_iterable(species))).tolist()
            species_counts = {split: {species_name: len(np.where(np.array(list(chain.from_iterable([species[idx] for idx in split_data[f'{split}_indices']]))) == species_name)[0]) for species_name in species_names} for split in all_splits}

        # calculate normalization statistics
        for split in subset_splits:
            for modality in ['Sentinel2', 'Sentinel1', 'AsterDEM', 'ETH_GCH', 'precipitation', 'temperature']:
                # train_images = h5_file[modality][:][split_data['train_indices']]
                train_images = h5_file[modality][:][split_data[f'{split}_indices']]
                masked = np.ma.masked_equal(train_images, no_data_values[modality])
                axes_to_collapse = tuple(i for i in range(masked.ndim) if i != 1) # collapse all dimensions except the channel dimension
                collapsed_shape = (masked.shape[1],) + (1,) * (masked.ndim - 2) # 0th dimension is the number of channels and singleton dimensions for the number of spatial dimensions
                # split_data[f'{modality}_train_means'] = masked.mean(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
                # split_data[f'{modality}_train_stds'] = masked.std(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
                split_data[f'{modality}_{split}_means'] = masked.mean(axis=axes_to_collapse).reshape(collapsed_shape).tolist()
                split_data[f'{modality}_{split}_stds'] = masked.std(axis=axes_to_collapse).reshape(collapsed_shape).tolist()

    # save split data
    with open(f'{data_dir_path}/{task}/{task}_split_data.json', 'w') as file:
        json.dump(split_data, file, indent=4)

    # save tile boxes for each split
    split_boxes = {split: [boxes[i] for i in split_data[f'{split}_indices']] for split in all_splits}
    os.makedirs(f'{data_dir_path}/{task}/split_tiles', exist_ok=True)

    for split in all_splits:
        gdf = gpd.GeoDataFrame({'index': split_data[f'{split}_indices'], 'geometry': split_boxes[split]}, crs='EPSG:4326')
        gdf.to_file(f'{data_dir_path}/{task}/split_tiles/{task}_{split}_tiles.geojson', driver='GeoJSON')

    if task != 'species':
        # calculate RMSE using the train mean as the prediction
        split_task_values = {split: task_data.squeeze()[split_data[f'{split}_indices']].ravel() for split in splits}

        if task == 'biomass':
            split_task_values = {split: [value for value in split_task_values[split] if value != -9999] for split in splits}

        train_mean = np.mean(split_task_values['train_100%'])
        val_rmse = np.sqrt(np.mean((np.array(split_task_values['val']) - train_mean) ** 2))
        random_test_rmse = np.sqrt(np.mean((np.array(split_task_values['random_test']) - train_mean) ** 2))
        geographic_test_rmse = np.sqrt(np.mean((np.array(split_task_values['geographic_test']) - train_mean) ** 2))

        # plot task distribution
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 16), sharex=True) # 4 rows, 1 column, shared x-axis
        max_value = max([np.max(split_task_values[split]) for split in splits])
        bin_size = 1
        bins = np.arange(0, max_value + bin_size, bin_size)
        tick_interval = np.ceil(max_value / 20)

        for i, split in enumerate(splits):
            counts, bin_edges = np.histogram(split_task_values[split], bins=bins)
            percentages = counts / counts.sum() * 100

            axes[i].bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), align='edge')
            axes[i].set_xticks(np.arange(0, max_value + tick_interval, tick_interval))
            axes[i].set_ylabel('Percentage (%)')
            axes[i].set_title(split.replace('train_100%', 'train').replace('_', ' ').capitalize())

        fig.suptitle(task.replace('_', ' ').capitalize().replace("ph", "pH"), fontweight='bold')
        axes[-1].set_xlabel(f'{task.replace("_", " ").capitalize().replace("ph", "pH")} value') # sets common x-label
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f'{data_dir_path}/{task}/{task}_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    # save summary for task
    with open(f'{data_dir_path}/{task}/output-files/{task}_summary.txt', 'w') as txt_file:
        txt_file.write(f'Task: {task}\n')
        txt_file.write(f'{task} tile count: {tile_count}\n')
        txt_file.write(f'{task}: {task_data.shape}\n')
        txt_file.write(f'Tiles in Africa: {len(africa_boxes)}\n')
        txt_file.write(f'Tiles outside Africa: {len(non_africa_boxes)}\n')
        txt_file.write(f'{len(split_data["train_100%_indices"])} training 100% tiles\n')
        txt_file.write(f'{len(split_data["train_50%_indices"])} training 50% tiles\n')
        txt_file.write(f'{len(split_data["train_5%_indices"])} training 5% tiles\n')
        txt_file.write(f'{len(split_data["val_indices"])} validation tiles\n')
        txt_file.write(f'{len(split_data["random_test_indices"])} random test tiles\n')
        txt_file.write(f'{len(split_data["geographic_test_indices"])} geographic test tiles\n')

        if task != 'species':
            txt_file.write(f'Mean of train values: {round(float(train_mean), 2)}\n')
            txt_file.write(f'STD of train values: {round(float(np.std(split_task_values["train_100%"])), 2)}\n')
            txt_file.write(f'Mean of validation values: {round(float(np.mean(split_task_values["val"])), 2)}\n')
            txt_file.write(f'STD of validation values: {round(float(np.std(split_task_values["val"])), 2)}\n')
            txt_file.write(f'Mean of random test values: {round(float(np.mean(split_task_values["random_test"])), 2)}\n')
            txt_file.write(f'STD of random test values: {round(float(np.std(split_task_values["random_test"])), 2)}\n')
            txt_file.write(f'Mean of geographic test values: {round(float(np.mean(split_task_values["geographic_test"])), 2)}\n')
            txt_file.write(f'STD of geographic test values: {round(float(np.std(split_task_values["geographic_test"])), 2)}\n')
            txt_file.write(f'Val RMSE using the train mean as the prediction: {round(float(val_rmse), 2)}\n')
            txt_file.write(f'Random test RMSE using the train mean as the prediction: {round(float(random_test_rmse), 2)}\n')
            txt_file.write(f'Geographic test RMSE using the train mean as the prediction: {round(float(geographic_test_rmse), 2)}\n')
        else:
            txt_file.write(f'Min count of a species in train 100%: {min(species_counts["train_100%"].values())}\n')
            txt_file.write(f'Min count of a species in train 50%: {min(species_counts["train_50%"].values())}\n')
            txt_file.write(f'Min count of a species in train 5%: {min(species_counts["train_5%"].values())}\n')
            txt_file.write(f'Min count of a species in val: {min(species_counts["val"].values())}\n')
            txt_file.write(f'Min count of a species in random test: {min(species_counts["random_test"].values())}\n')
            txt_file.write(f'Min count of a species in geographic test: {min(species_counts["geographic_test"].values())}\n')

    # plot dataset split on world map
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global() # shows the whole world
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)

    split_properties = {'train_100%': {'color': 'blue', 'label': 'Training'},
                        'val': {'color': 'green', 'label': 'Validation'},
                        'random_test': {'color': 'orange', 'label': 'Random test'},
                        'geographic_test': {'color': 'red', 'label': 'Geographic test'}}

    for split in splits:
        gdf = gpd.read_file(f'{data_dir_path}/{task}/split_tiles/{task}_{split}_tiles.geojson').to_crs(epsg=3857)
        gdf['geometry'] = gdf.geometry.centroid.to_crs(epsg=4326)
        gdf.plot(ax=ax, color=split_properties[split]['color'], marker='s', markersize=20, label=split_properties[split]['label'])

    ax.legend(handles=[Line2D([0], [0], marker='s', color='w', markerfacecolor=split_properties[mode]['color'], markersize=15, label=split_properties[mode]['label']) for mode in split_properties.keys()],
              loc='lower left',
              fontsize=12)
    plt.title(f'{task.replace("_", " ").title().replace("Ph", "pH")} Dataset Split', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{data_dir_path}/{task}/{task}_split_map.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    generate_splits(task=argv[1])
