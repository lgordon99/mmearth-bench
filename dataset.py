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
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
training_fraction = 0.7
validation_fraction = 0.15
random.seed(42)

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
    def __init__(self, task, split_type):
        self.task = task

        with h5py.File(f'{data_dir_path}/{task}/{task}_h5.hdf5', 'r') as h5_file:
            self.tile_ids = h5_file['id'][:]
            self.tile_count = len(self.tile_ids) # number of tiles for the task
            self.sentinel2 = h5_file['Sentinel2'][:]
            self.task_data = h5_file[task][:]
            self.crs = h5_file['crs'][:].astype(str).tolist() # coordinate reference system for each tile
            self.transforms = h5_file['transform'][:] # affine transformation for each tile
            self.split_data_path = f'{split_type}_split_data.json'

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
        height, width = self.sentinel2.shape[-2:] # height and width of the Sentinel-2 images

        if split_type == 'world_random': # for random split over the whole world
            tile_ids = self.tile_ids.copy()
            random.shuffle(tile_ids) # randomly reorders the tile IDs
            end_train_ids = int(training_fraction * self.tile_count) # 70% of the data for training
            end_val_ids = int((training_fraction+validation_fraction) * self.tile_count) # 15% of the data for validation

            train_ids = tile_ids[:end_train_ids].tolist()
            val_ids = tile_ids[end_train_ids:end_val_ids].tolist()
            test_ids = tile_ids[end_val_ids:].tolist()
        else:
            country_data = utils.read_geojson(f'{data_dir_path}/world_administrative_boundaries.geojson')['features'] # country boundary data
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
            boxes = {self.tile_ids[i]: get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in range(self.tile_count)} # dictionairy of boxes for each tile
            africa_boxes = {tile_id: box for tile_id, box in boxes.items() if box.within(africa_boundaries)} # dictionairy of boxes for each tile within the Africa boundaries
            non_africa_boxes = {tile_id: box for tile_id, box in boxes.items() if box.disjoint(africa_boundaries)}
            print(f'Tiles in Africa: {len(africa_boxes)}')
            print(f'Tiles outside Africa: {len(non_africa_boxes)}')

            # # plot the Africa split
            # fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            # ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            # ax.add_feature(cfeature.COASTLINE, linewidth=1)
            # ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7)
            # ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
            # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            # africa_gdf = gpd.GeoDataFrame(geometry=[africa_boundaries])
            # africa_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1, transform=ccrs.PlateCarree())
            # africa_boxes_gdf = gpd.GeoDataFrame(geometry=list(africa_boxes.values()))
            # africa_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=1, transform=ccrs.PlateCarree())
            # non_africa_boxes_gdf = gpd.GeoDataFrame(geometry=list(non_africa_boxes.values()))
            # non_africa_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1, transform=ccrs.PlateCarree())
            # ax.set_title(f'{self.task} Africa Split', fontsize=14)
            # plt.savefig(f'figures/{self.task}_africa_split.png', dpi=300, bbox_inches='tight')
            # plt.close(fig)

            # training and validation tile IDs
            non_africa_tile_ids = list(map(int, non_africa_boxes.keys()))
            random.shuffle(non_africa_tile_ids) # randomly reorders the non-Africa-tile IDs
            end_train_ids = int(training_fraction * len(non_africa_tile_ids)) # 70% of the non-Africa tiles for training
            end_val_ids = int((training_fraction+validation_fraction) * len(non_africa_tile_ids)) # 15% of the data for validation
            train_ids = non_africa_tile_ids[:end_train_ids]
            val_ids = non_africa_tile_ids[end_train_ids:end_val_ids]

            if split_type == 'random':
                test_ids = non_africa_tile_ids[end_val_ids:]
            elif split_type == 'geographic':
                test_ids = list(map(int, africa_boxes.keys()))

        print(f'Dataset lengths for {split_type} split: Train = {len(train_ids)}, Val = {len(val_ids)}, Test = {len(test_ids)}')
        train_indices = np.array([np.where(self.tile_ids == tile_id)[0][0] for tile_id in train_ids])
        train_images = self.sentinel2[train_indices]
        self.train_band_means = train_images.mean(axis=(0,2,3))[:, None, None].tolist()
        self.train_band_stds = train_images.std(axis=(0,2,3))[:, None, None].tolist()
        split_data = {'train_ids': train_ids,
                      'val_ids': val_ids,
                      'test_ids': test_ids,
                      'train_band_means': self.train_band_means,
                      'train_band_stds': self.train_band_stds}

        with open(f'{split_type}_split_data.json', 'w') as file:
            json.dump(split_data, file, indent=4)

        val_indices = np.array([np.where(self.tile_ids == tile_id)[0][0] for tile_id in val_ids])
        test_indices = np.array([np.where(self.tile_ids == tile_id)[0][0] for tile_id in test_ids])
        train_boxes = {self.tile_ids[i]: get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in train_indices} # dictionairy of boxes for the training tiles
        val_boxes = {self.tile_ids[i]: get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in val_indices}
        test_boxes = {self.tile_ids[i]: get_box_wgs_84(transform=self.transforms[i], width=width, height=height, crs=self.crs[i]) for i in test_indices}

        # plot the split
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        train_boxes_gdf = gpd.GeoDataFrame(geometry=list(train_boxes.values()))
        train_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1, transform=ccrs.PlateCarree())
        val_boxes_gdf = gpd.GeoDataFrame(geometry=list(val_boxes.values()))
        val_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=1, transform=ccrs.PlateCarree())
        test_boxes_gdf = gpd.GeoDataFrame(geometry=list(test_boxes.values()))
        test_boxes_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1, transform=ccrs.PlateCarree())
        legend_handles = [Line2D([0], [0], color='red', lw=2, label='Train'),
                          Line2D([0], [0], color='green', lw=2, label='Val'),
                          Line2D([0], [0], color='blue', lw=2, label='Test')]
        ax.legend(handles=legend_handles, loc='upper right')
        ax.set_title(f'{self.task} {split_type} Split', fontsize=14)
        plt.savefig(f'figures/{self.task}_{split_type}_split.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def __len__(self):
        return self.tile_count

    def __getitem__(self, tile_id):
        index = np.where(self.tile_ids == tile_id)[0][0] # gets the index of the tile_id
        sentinel2 = self.sentinel2[index]
        sentinel2 = (sentinel2 - self.train_band_means) / self.train_band_stds # normalization
        task_data = self.task_data[index]

        return sentinel2, task_data
