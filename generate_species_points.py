# REQUIRES 10GB RAM

# ============================================== IMPORTS ============================================== #

from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm import tqdm
import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

random.seed(42) # sets a seed for reproducibility
ee.Initialize(project='mmearth-bench') # initializes EE with the project
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

# def get_africa_boundaries():
#     continents = utils.read_geojson(f'{data_dir_path}/World_Continents_-8423622284373332078.geojson')['features']
#     africa = [continent for continent in continents if continent['properties']['CONTINENT'] == 'Africa'][0]
#     tolerance = 9
#     smoothed_africa = shape(africa['geometry']).buffer(tolerance).buffer(-tolerance) # smooths the boundaries
#     gpd.GeoDataFrame(geometry=[smoothed_africa], crs='EPSG:4326').to_file(f'{data_dir_path}/africa_9.geojson', driver='GeoJSON') # saves the Africa boundaries as a GeoJSON file

    # country_data = utils.read_geojson(f'{data_dir_path}/world_administrative_boundaries.geojson')['features'] # country boundary data
    # african_country_data = [country for country in country_data if country['properties'].get('continent') == 'Africa'] # African country boundary data
    # africa_polygons = [] # list of polygons for boundaries of African countries

    # for country in african_country_data: # for each African country
    #     geometry = country['geometry'] # extracts the country's geometry

    #     if geometry['type'] == 'Polygon': # if the geometry is a polygon
    #         africa_polygons.append(Polygon(geometry['coordinates'][0])) # saves the polygon's coordinates
    #     elif geometry['type'] == 'MultiPolygon': # if the geometry is multiple polygons
    #         for polygon_coordinates in geometry['coordinates']: # for each polygon
    #             africa_polygons.append(Polygon(polygon_coordinates[0])) # saves the polygon's coordinates

    # tolerance = 0.1
    # africa_boundaries = MultiPolygon(africa_polygons).buffer(tolerance).buffer(-tolerance) # boundaries of all the African countries
    # gpd.GeoDataFrame(geometry=[africa_boundaries], crs='EPSG:4326').to_file(f'{data_dir_path}/africa.geojson', driver='GeoJSON') # saves the Africa boundaries as a GeoJSON file

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

    # mammals_in_africa = mammals_in_africa.sort_values(by='area_outside_africa', ascending=False)[:100] # selects species with the largest area outside Africa
    # print(f'Min area in Africa: {mammals_in_africa["area_in_africa"].min()}, Min area outside Africa: {mammals_in_africa["area_outside_africa"].min()}')
    # mammals_in_africa.to_file(f'{data_dir_path}/species/species_ranges.geojson', driver='GeoJSON')

    # os.makedirs(f'{data_dir_path}/species/species_ranges', exist_ok=True)

    # for _, row in mammals_in_africa.iterrows():
    #     species_name = row["sci_name"].replace(" ", "_").replace("/", "_")
    #     gdf_row = gpd.GeoDataFrame([row], columns=mammals_in_africa.columns, crs=mammals_in_africa.crs)
    #     gdf_row.to_file(f"{data_dir_path}/species/species_ranges/{species_name}.geojson", driver="GeoJSON")

    # top_in_africa = mammals_in_africa.sort_values(by='area_in_africa', ascending=False)[:1100] # selects species with the largest area in Africa
    # top_outside_africa = mammals_in_africa.sort_values(by='area_outside_africa', ascending=False)[:100] # selects species with the largest area outside Africa
    # mammals_top = mammals_in_africa[mammals_in_africa['sci_name'].isin(top_in_africa['sci_name']) & mammals_in_africa['sci_name'].isin(top_outside_africa['sci_name'])] # selects species in both top lists
    # print(f'{len(mammals_top)} species in both top lists')
    # mammals_top = mammals_top[:90] # limits to top 90 species in both
    # print(f'Min area in Africa: {mammals_top["area_in_africa"].min()}, Min area outside Africa: {mammals_top["area_outside_africa"].min()}')
    # os.makedirs(f'{data_dir_path}/species/species_ranges', exist_ok=True)
    # mammals_top.to_file(f'{data_dir_path}/species/species_ranges/species_ranges.shp', driver='ESRI Shapefile') # saves the top species ranges as a shapefile

def generate_species_points():
    start_time = time.time()
    species_gdf = gpd.read_file(f'{data_dir_path}/species/species_ranges/species_ranges.shp')
    OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']
    africa_union = unary_union(gpd.read_file(f'{data_dir_path}/africa.geojson').geometry) # union of all African country boundaries
    points = []

    for row in tqdm(species_gdf.itertuples(index=False), total=len(species_gdf), desc='Species'):
        species = row.sci_name
        range_in_africa = row.geometry.intersection(africa_union)
        range_outside_africa = row.geometry.difference(africa_union)
        split_data = {'in_africa': {'num_tiles': 100, 'range': range_in_africa, 'points': []},
                      'outside_africa': {'num_tiles': 300, 'range': range_outside_africa, 'points': []}}

        for split, data in split_data.items():
            minx, miny, maxx, maxy = data['range'].bounds # bounding box of the range

            while len(split_data[split]['points']) < data['num_tiles']:
                x = random.uniform(minx, maxx) # random x coordinate within the bounding box
                y = random.uniform(miny, maxy) # random y coordinate within the bounding box
                # point = Point(x, y) # creates a point from the coordinates
                point = ee.Feature(ee.Geometry.Point([x, y])).set({'species': [species], 'outer_tile': ee.Geometry.Point([x, y]).buffer(OUTER_TILE_SIZE_M / 2).bounds()}).getInfo()

                # if data['range'].contains(point): # if the point is within the range
                if data['range'].intersects(shape(point['properties']['outer_tile'])): # if the point's outer tile intersects the range
                    # split_data[split]['points'].append(ee.Feature(ee.Geometry.Point([x, y])).set({'species': [species], 'outer_tile': ee.Geometry.Point([x, y]).buffer(OUTER_TILE_SIZE_M / 2).bounds()}).getInfo()) # creates an outer tile around each point
                    split_data[split]['points'].append(point)

        species_points = [point for split in split_data.keys() for point in split_data[split]['points']]
        print(f'{len(species_points)} before removing overlaps')
        species_points = utils.remove_overlapping_tiles(species_points) # removes points with overlapping tiles
        print(f'{len(species_points)} points after removing overlaps')
        points.extend(species_points)

    print(f'{len(points)} before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    for point in tqdm(points, total=len(points), desc='Points'):
        for row in species_gdf.itertuples(index=False): # for each species
            if row.sci_name not in point['properties']['species'] and row.geometry.intersects(shape(point['properties']['outer_tile'])): # if the point's outer tile intersects the species range
                point['properties']['species'].append(row.sci_name) # adds the species to the point's species list

    random.shuffle(points) # shuffles the points list
    points = [{**{key: value for key, value in point.items() if key != 'properties'}, 'properties': {key: value for key, value in point['properties'].items() if key != 'outer_tile'}, 'id': i} for i, point in enumerate(points)] # assigns each point an ID and removes the outer tile
    utils.save_geojson(features=points, path=f'{data_dir_path}/species/species_points.geojson') # saves the points as a GeoJSON

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_point_statistics():
    points = utils.read_geojson(f'{data_dir_path}/species/species_points.geojson')['features']
    print(f'Total points: {len(points)}')

    # plot number of points per species
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
    plt.ylabel('Number of points')
    plt.title('Number of points per species')
    plt.tight_layout()
    plt.savefig(f'{data_dir_path}/species/points_per_species.png')

    print(f'Max number of points for a species: {max(counts)}')
    print(f'Min number of points for a species: {min(counts)}')

    # histogram of number of species per point
    point_species_counts = []

    for point in points:
        point_species_counts.append(len(point['properties']['species']))

    plt.figure(dpi=300, figsize=(15, 5))
    plt.hist(point_species_counts, bins=np.arange(min(point_species_counts), max(point_species_counts) + 2) - 0.5)
    plt.xlabel('Number of species')
    plt.ylabel('Number of points')
    plt.tight_layout()
    plt.savefig(f'{data_dir_path}/species/point_species_counts.png')

if __name__ == '__main__':
    # if not os.path.exists(f'{data_dir_path}/africa.geojson'):
    #     get_africa_boundaries()

    if not os.path.exists(f'{data_dir_path}/species/species_ranges.geojson'):
        get_species_ranges()

    if not os.path.exists(f'{data_dir_path}/species/species_points.geojson'):
        generate_species_points()

    check_point_statistics()
