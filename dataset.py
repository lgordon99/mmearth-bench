'''
dataset.py
'''

# ============================================== IMPORTS ============================================== #

from affine import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import box, Polygon, MultiPolygon
from torch.utils.data import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import h5py
import json
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import utils

# ============================================== FUNCTIONS ============================================== #

def get_box_wgs_84(transform, width, height, crs):
    affine_transform = Affine(*transform.tolist())
    left, top = affine_transform * (0, 0)
    right, bottom = affine_transform * (width, height)
    bounds = (left, bottom, right, top)

    if crs != 'EPSG:4326':
        bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

    return box(*bounds)

# ============================================== CLASSES ============================================== #

class MMEarthBenchDataset(Dataset):
    def __init__(self, task, split_type, data_dir_path):
        self.task = task
        self.split_type = split_type
        self.data_dir_path = data_dir_path

        # with h5py.File(f'{data_dir_path}/{task}/{task}_h5.hdf5', 'r') as h5_file:
        with h5py.File(f'/scratch/{task}_h5.hdf5', 'r') as h5_file:
            self.tile_ids = h5_file['id'][:]
            self.tile_count = len(self.tile_ids) # number of tiles for the task
            self.sentinel2 = h5_file['Sentinel2'][:]
            self.task_data = h5_file[task][:]
            self.crs = h5_file['crs'][:].astype(str).tolist() # coordinate reference system for each tile
            self.transforms = h5_file['transform'][:] # affine transformation for each tile
            # self.split_data_path = f'{data_dir_path}/{task}/{task}_{split_type}_split_data.json'
            self.split_data_path = f'/scratch/{task}_{split_type}_split_data.json'

            print(f'{task} tile count: {self.tile_count}')
            print(f'Sentinel-2: {self.sentinel2.shape}')
            print(f'{task}: {self.task_data.shape}')

        if os.path.exists(self.split_data_path): # if the split data has already been generated
            split_data = utils.read_json(self.split_data_path)
            self.train_band_means = split_data['train_band_means']
            self.train_band_stds = split_data['train_band_stds']
        else: # if the split data has not been generated
            self._get_split_data(split_type)

    def _get_split_data(self, split_type):
        training_fraction = 0.7
        validation_fraction = 0.15
        height, width = self.sentinel2.shape[-2:] # height and width of the Sentinel-2 images

        if split_type == 'world_random': # for random split over the whole world
            tile_indices = list(range(self.tile_count))
            random.shuffle(tile_indices) # randomly reorders the tile IDs
            end_train_indices = int(training_fraction * self.tile_count) # 70% of the data for training
            end_val_indices = int((training_fraction+validation_fraction) * self.tile_count) # 15% of the data for validation

            train_indices = tile_indices[:end_train_indices]
            val_indices = tile_indices[end_train_indices:end_val_indices]
            test_indices = tile_indices[end_val_indices:]
        else:
            country_data = utils.read_geojson(f'{self.data_dir_path}/world_administrative_boundaries.geojson')['features'] # country boundary data
            african_country_data = [country for country in country_data if country['properties'].get('continent') == 'Africa'] # African country boundary data
            africa_polygons = [] # list of polygons for boundaries of African countries

            for country in african_country_data: # for each African country
                geometry = country['geometry'] # extracts the country's geometry

                if geometry['type'] == 'Polygon': # if the geometry is a polygon
                    africa_polygons.append(Polygon(geometry['coordinates'][0])) # saves the polygon's coordinates
                elif geometry['type'] == 'MultiPolygon': # if the geometry is multiple polygons
                    for polygon_coordinates in geometry['coordinates']: # for each polygon
                        africa_polygons.append(Polygon(polygon_coordinates[0])) # saves the polygon's coordinates

            africa_boundaries = MultiPolygon(africa_polygons) # boundaries of all the African countries
            boxes = {i: get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in range(self.tile_count)} # dictionairy of boxes for each tile
            africa_boxes = {tile_index: box for tile_index, box in boxes.items() if box.within(africa_boundaries)} # dictionairy of boxes for each tile within the Africa boundaries
            non_africa_boxes = {tile_index: box for tile_index, box in boxes.items() if box.disjoint(africa_boundaries)}
            print(f'Tiles in Africa: {len(africa_boxes)}')
            print(f'Tiles outside Africa: {len(non_africa_boxes)}')

            # training and validation tile IDs
            non_africa_tile_indices = list(map(int, non_africa_boxes.keys()))
            random.shuffle(non_africa_tile_indices) # randomly reorders the non-Africa-tile indices
            end_train_indices = int(training_fraction * len(non_africa_tile_indices)) # 70% of the non-Africa tiles for training
            end_val_indices = int((training_fraction+validation_fraction) * len(non_africa_tile_indices)) # 15% of the non-Africa tiles for validation
            train_indices = non_africa_tile_indices[:end_train_indices]
            val_indices = non_africa_tile_indices[end_train_indices:end_val_indices]

            if split_type == 'random':
                test_indices = non_africa_tile_indices[end_val_indices:] # remaining 15% of the non-Africa tiles for testing
            elif split_type == 'geographic':
                test_indices = list(map(int, africa_boxes.keys())) # Africa tiles for testing

        if self.task == 'species':
            train_species_counts = self.task_data[train_indices].sum(axis=0)
            species_with_too_few_train_observations = np.where(train_species_counts < 50)[0]
            self.task_data[np.ix_(test_indices, species_with_too_few_train_observations)] = 0

        print(f'Dataset lengths for {split_type} split: Train = {len(train_indices)}, Val = {len(val_indices)}, Test = {len(test_indices)}')
        train_images = self.sentinel2[train_indices]
        self.train_band_means = train_images.mean(axis=(0,2,3))[:, None, None].tolist()
        self.train_band_stds = train_images.std(axis=(0,2,3))[:, None, None].tolist()
        split_data = {'train_indices': train_indices,
                      'val_indices': val_indices,
                      'test_indices': test_indices,
                      'train_band_means': self.train_band_means,
                      'train_band_stds': self.train_band_stds}

        with open(f'{self.data_dir_path}/{self.task}/{self.task}_{self.split_type}_split_data.json', 'w') as file:
            json.dump(split_data, file, indent=4)

        with open(self.split_data_path, 'w') as file:
            json.dump(split_data, file, indent=4)

        train_boxes = [get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in train_indices] # dictionairy of boxes for the training tiles
        val_boxes = [get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in val_indices]
        test_boxes = [get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in test_indices]

        # plot the split
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        train_boxes_gdf = gpd.GeoDataFrame(geometry=train_boxes)
        train_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1, transform=ccrs.PlateCarree())
        val_boxes_gdf = gpd.GeoDataFrame(geometry=val_boxes)
        val_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=1, transform=ccrs.PlateCarree())
        test_boxes_gdf = gpd.GeoDataFrame(geometry=test_boxes)
        test_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1, transform=ccrs.PlateCarree())
        legend_handles = [Line2D([0], [0], color='red', lw=2, label='Train'),
                          Line2D([0], [0], color='green', lw=2, label='Val'),
                          Line2D([0], [0], color='blue', lw=2, label='Test')]
        ax.legend(handles=legend_handles, loc='upper right')
        ax.set_title(f'{self.task} {split_type} Split', fontsize=14)
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{self.task}_{split_type}_split.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def __len__(self):
        return self.tile_count

    def __getitem__(self, index):
        sentinel2 = self.sentinel2[index]
        sentinel2 = (sentinel2 - self.train_band_means) / self.train_band_stds # normalization
        task_data = self.task_data[index]

        if len(task_data.shape) == 2:
            task_data = np.expand_dims(task_data, axis=0)

        return torch.tensor(sentinel2, dtype=torch.float32), torch.tensor(task_data, dtype=torch.float32)
