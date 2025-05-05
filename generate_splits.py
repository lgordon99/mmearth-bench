# ============================================== IMPORTS ============================================== #

from affine import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import box, Polygon, MultiPolygon
import geopandas as gpd
import h5py
import json
import random
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

training_fraction = 0.7
validation_fraction = 0.15
tolerance = 0.1
height, width = 128, 128
data_dir_path = '/n/davies_lab/Users/luciagordon/mmearth-bench'
random.seed(42) # for reproducibility

# ============================================== FUNCTIONS ============================================== #

def get_box_wgs_84(transform, width, height, crs):
    affine_transform = Affine(*transform.tolist())
    left, top = affine_transform * (0, 0)
    right, bottom = affine_transform * (width, height)
    bounds = (left, bottom, right, top)

    if crs != 'EPSG:4326':
        bounds = transform_bounds(crs, 'EPSG:4326', *bounds)

    return box(*bounds)

def get_africa_boundaries():
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

    africa_boundaries = MultiPolygon(africa_polygons).buffer(tolerance).buffer(-tolerance) # boundaries of all the African countries
    africa_gdf = gpd.GeoDataFrame(geometry=[africa_boundaries], crs='EPSG:4326')
    africa_gdf.to_file(f'{data_dir_path}/{task}/{task}_africa.geojson', driver='GeoJSON')

    return africa_boundaries

def get_split_data(task):
    with h5py.File(f'{data_dir_path}/{task}/{task}.h5', 'r') as h5_file:
        sentinel2 = h5_file['Sentinel2'][:]
        task_data = h5_file[task][:]
        crs = h5_file['crs'][:].astype(str).tolist() # coordinate reference system for each tile
        transforms = h5_file['transform'][:] # affine transformation for each tile
        tile_count = len(sentinel2) # number of tiles for the task
        split_data_path = f'{data_dir_path}/{task}/{task}_split_data_new.json'

        print(f'{task} tile count: {tile_count}')
        print(f'Sentinel-2: {sentinel2.shape}')
        print(f'{task}: {task_data.shape}')

    boxes = {i: get_box_wgs_84(transform=transforms[i], width=width, height=height, crs=crs[i]) for i in range(tile_count)} # dictionary of boxes for each tile
    africa_boundaries = get_africa_boundaries()
    africa_boxes = {tile_index: box for tile_index, box in boxes.items() if box.intersects(africa_boundaries)} # dictionary of boxes for each tile within the Africa boundaries
    non_africa_boxes = {tile_index: box for tile_index, box in boxes.items() if tile_index not in africa_boxes.keys()} # dictionary of boxes for each tile outside the Africa boundaries
    print(f'Tiles in Africa: {len(africa_boxes)}')
    print(f'Tiles outside Africa: {len(non_africa_boxes)}')

    # training, validation, and testing tile indices
    non_africa_tile_indices = sorted(list(map(int, non_africa_boxes.keys())))
    random.shuffle(non_africa_tile_indices) # randomly reorders the non-Africa-tile indices
    end_train_indices = int(training_fraction * len(non_africa_tile_indices)) # 70% of the non-Africa tiles for training
    end_val_indices = int((training_fraction+validation_fraction) * len(non_africa_tile_indices)) # 15% of the non-Africa tiles for validation
    split_data = {}
    split_data['train_indices'] = sorted(non_africa_tile_indices[:end_train_indices])
    split_data['val_indices'] = sorted(non_africa_tile_indices[end_train_indices:end_val_indices])
    train_images = sentinel2[split_data['train_indices']]
    split_data['train_band_means'] = train_images.mean(axis=(0,2,3))[:, None, None].tolist()
    split_data['train_band_stds'] = train_images.std(axis=(0,2,3))[:, None, None].tolist()
    split_data['random_test_indices'] = sorted(non_africa_tile_indices[end_val_indices:]) # remaining 15% of the non-Africa tiles for testing
    split_data['geographic_test_indices'] = sorted(list(map(int, africa_boxes.keys()))) # Africa tiles for testing

    with open(split_data_path, 'w') as file:
        json.dump(split_data, file, indent=4)

    print(f'{len(split_data["train_indices"])} training tiles')
    print(f'{len(split_data["val_indices"])} validation tiles')
    print(f'{len(split_data["random_test_indices"])} random test tiles')
    print(f'{len(split_data["geographic_test_indices"])} geographic test tiles\n')

    split_boxes = {split: [boxes[i] for i in split_data[f'{split}_indices']] for split in ['train', 'val', 'random_test', 'geographic_test']}

    for split in ['train', 'val', 'random_test', 'geographic_test']:
        gpd.GeoDataFrame(geometry=split_boxes[split], crs='EPSG:4326').to_file(f'{data_dir_path}/{task}/{task}_{split}_tiles.geojson', driver='GeoJSON')

if __name__ == '__main__':
    for task in ['biomass', 'species', 'soil_nitrogen', 'soil_organic_carbon', 'soil_pH']:
        get_split_data(task)
