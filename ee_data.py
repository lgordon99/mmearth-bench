'''
ee_data.py
A general class to collect the data from GEE. Each function in the class will be a different dataset, and they share common variables like the start and end date, the projection etc.
'''

# imports
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shapely.geometry import Polygon
import certifi
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import time
import utils

TILE_SIZE_M = utils.read_yaml('config.yml')['TILE_SIZE_M']
TILE_SIZE = int(TILE_SIZE_M / 10)
year = '2020'

def get_last_day_of_month(month):
    return (datetime(int(year), month, 1) + relativedelta(months=1, days=-1)).day

class EEData:
    def __init__(self, point, task, data_dir_path):
        start = time.time()
        self.point = point
        self.bands = []
        self.task = task
        self.gedi_points = self.get_gedi_points() if task == 'biomass' else None
        self.date_filter = ee.Filter.Or(*[ee.Filter.date(date_range[0], date_range[1]) for date_range in self.get_dates()])
        self.crs = ''
        self.s2_date = ''
        self.month_encoding = ''
        self.id = point['id']
        self.tile = ee.Geometry.Point(point['geometry']['coordinates']).buffer(TILE_SIZE_M / 2).bounds()
        self.lon, self.lat = point['geometry']['coordinates']
        self.geolocation_encoding = {'lat_sin': np.sin(np.deg2rad(self.lat)),
                                     'lat_cos': np.cos(np.deg2rad(self.lat)),
                                     'lon_sin': np.sin(np.deg2rad(self.lon)),
                                     'lon_cos': np.cos(np.deg2rad(self.lon))}
        self.pixel_level_data = {}
        self.no_data = False
        self.era5_data = {}
        self.proj = ''
        self.task_values = None
        self.missing_modalities = []
        self.data_dir_path = data_dir_path

        if 'biome' in list(point['properties'].keys()):
            self.name_biome = point['properties']['biome']
            self.name_ecoregion = point['properties']['ecoregion']
        else:
            ecoregion_features = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.tile).getInfo()['features']

            if len(ecoregion_features) > 0:
                self.name_biome = ecoregion_features[0]['properties']['BIOME_NAME']
                self.name_ecoregion = ecoregion_features[0]['properties']['ECO_NAME']

            if len(ecoregion_features) == 0 or self.name_biome == 'N/A':
                self.name_biome = None
                self.name_ecoregion = None
                self.missing_modalities.append('biome/ecoregion')

        if self.name_biome is not None:
            biome_labels = utils.read_json('biomes_ecoregions_data/biome_labels.json')
            ecoregion_labels = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')

            self.biome = biome_labels[self.name_biome]
            self.ecoregion = ecoregion_labels[self.name_ecoregion]
        else:
            self.biome = None
            self.ecoregion = None

        datasets = ['sentinel2', 'sentinel1', 'aster', 'canopy_height_eth', 'dynamic_world', 'esa_worldcover', 'era5']

        for function_name in datasets: # series of function calls to get the data
            if getattr(self, function_name)() is False: # if the method returns False
                self.missing_modalities.append(function_name)
                print(f'{function_name} is missing')

                if function_name == 'sentinel2':
                    self.no_data = True
                    break

        if not self.no_data:
            self.task_data(task)

            merged_image = self.pixel_level_data[datasets[0]] # start with Sentinel-2

            for dataset, value in self.pixel_level_data.items():
                if dataset == datasets[0]:
                    continue # already included Sentinel-2
                else:
                    merged_image = ee.Image.cat([merged_image, value])

            self.save_tiff(merged_image)

            print(f'Time elapsed for this tile: {round(time.time() - start, 2)}s')

    def get_gedi_points(self):
        collection_names = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                            .filter(f'time_start >= "{year}-01-01" && time_end <= "{year}-12-31"') # get feature collections with features in the selected year
                            .filterBounds(self.point['properties']['outer_tile']) # get feature collections that have features within the tile
                            .aggregate_array('table_id') # extract the IDs of the feature collections
                            .getInfo()) # list of names of the feature collections
        quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
        gedi_points = (ee.FeatureCollection([result for _, result in ((collection_name, utils.get_asset_if_valid(ee.FeatureCollection(collection_name))) for collection_name in collection_names) if result is not None])
                        .flatten() # merge all the feature collections into one
                        .filterBounds(self.point['properties']['outer_tile']) # collection of the features that are within the tile
                        .filter(quality_filter) # apply the quality filter
                        .map(lambda point: point.set('off_after_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
                        .filter(ee.Filter.gt('off_after_on', 0))) # filtering by GEDI points with leaf on before leaf off

        return gedi_points

    def get_dates(self):
        if self.task == 'biomass':
            leaf_on_off = np.array([self.gedi_points.aggregate_array('leaf_on_doy').getInfo(), self.gedi_points.aggregate_array('leaf_off_doy').getInfo()]).T # get pairs of leaf on and off days for each point
            leaf_on_off_unique = list(map(list, set(map(tuple, leaf_on_off)))) # get unique pairs
            leaf_on_off_dates = [[pd.to_datetime(pair[0], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d'), pd.to_datetime(pair[1], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d')] for pair in leaf_on_off_unique]

            return leaf_on_off_dates

        elif self.task == 'species':
            month = self.point['properties']['month']

            if month > 1 and month < 12:
                start_month = month - 1
                end_month = month + 1
                dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]
            elif month == 1:
                start_month = 12
                end_month = 2
                dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(start_month).zfill(2)}-31'], [f'{year}-{str(month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]
            elif month == 12:
                start_month = 11
                end_month = 1
                dates = [[f'{year}-{str(start_month).zfill(2)}-01', f'{year}-{str(month).zfill(2)}-31'], [f'{year}-{str(end_month).zfill(2)}-01', f'{year}-{str(end_month).zfill(2)}-{get_last_day_of_month(end_month)}']]

            return dates

        elif 'soil' in self.task:
            point_latitude = self.point['geometry']['coordinates'][1]

            if point_latitude > 0:
                dates = [[f'{year}-{str(5).zfill(2)}-01', f'{year}-{str(9).zfill(2)}-{get_last_day_of_month(9)}']]
            elif point_latitude < 0:
                dates = [[f'{year}-{str(11).zfill(2)}-01', f'{year}-{str(12).zfill(2)}-{get_last_day_of_month(12)}'], [f'{year}-{str(1).zfill(2)}-01', f'{year}-{str(3).zfill(2)}-{get_last_day_of_month(3)}']]

            return dates

    def sentinel2(self):
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B11', 'B12', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL', 'QA60']
        sentinel2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') # L2A collection
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.tile) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.tile.buffer(200)))) # gets images containing the tile plus some buffer

        if sentinel2_images.size().getInfo() == 0: # if no images passed the date and location filters
            return False

        # print('Number of images before cloud filtering =', sentinel2_images.size().getInfo())

        msk_cldprob_res = sentinel2_images.select('MSK_CLDPRB').first().projection().nominalScale().getInfo() # resolution of MSK_CLDPRB band
        sentinel2_images = sentinel2_images.map(lambda image: image.set('MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', image.select('MSK_CLDPRB').gte(10).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=msk_cldprob_res).get('MSK_CLDPRB'))).filter(ee.Filter.notNull(['MSK_CLDPRB_CLOUDY_PIXEL_FRACTION'])) # pixels >= 10% cloud probability are cloudy
        sentinel2_images = sentinel2_images.filterMetadata('MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', 'less_than', 0.1) # keeps images with less than 10% cloudy pixels

        if sentinel2_images.size().getInfo() == 0: # if no images passed the MSK_CLDPRB filter
            return False

        # print('Number of images after applying the MSK_CLDPRB filter =', sentinel2_images.size().getInfo())

        s2cloudless_images = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                                .filter(self.date_filter)
                                .filterBounds(self.tile)
                                .filter(ee.Filter.contains('.geo', self.tile.buffer(200))))
        sentinel2_images = ee.ImageCollection(ee.Join.saveFirst('S2CLOUDLESS').apply(**{'primary': sentinel2_images, 'secondary': s2cloudless_images, 'condition': ee.Filter.equals(**{'leftField': 'system:index', 'rightField': 'system:index'})}))
        sentinel2_images = sentinel2_images.map(lambda image: image.addBands(ee.Image(image.get('S2CLOUDLESS')).rename('S2CLOUDLESS')))
        s2cloudless_res = s2cloudless_images.select('probability').first().projection().nominalScale().getInfo() # resolution of s2cloudless probability band
        sentinel2_images = sentinel2_images.map(lambda image: image.set('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', image.select('S2CLOUDLESS').gte(10).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=s2cloudless_res).get('S2CLOUDLESS'))) # pixels >= 10% cloud probability are cloudy
        sentinel2_images = sentinel2_images.filterMetadata('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', 'less_than', 0.1).sort('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION') # keeps images with less than 10% cloudy pixels

        # print('Number of images after applying the S2CLOUDLESS filter =', sentinel2_images.size().getInfo())

        if sentinel2_images.size().getInfo() == 0: # if no images passed the S2CLOUDLESS filter
            return False

        # print(sentinel2_images.aggregate_array('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION').getInfo())

        s2_image = sentinel2_images.first().float() # get the least cloudy S2 image
        self.s2_date = s2_image.date().format('YYYY-MM-dd').getInfo() # date of S2 image
        month = int(self.s2_date.split('-')[1])
        self.month_encoding = {'month_sin': np.sin(np.pi * month / 6), 'month_cos': np.cos(np.pi * month / 6)}
        self.proj = s2_image.select('B4').projection() # projection of B4 band
        projected_point_coordinates = ee.Geometry.Point(self.point['geometry']['coordinates']).transform(self.proj).coordinates()
        nearest_pixel_intersection_x = ee.Number(projected_point_coordinates.get(0)).round()
        nearest_pixel_intersection_y = ee.Number(projected_point_coordinates.get(1)).round()
        self.tile = ee.Geometry.Rectangle([nearest_pixel_intersection_x.subtract(TILE_SIZE/2), nearest_pixel_intersection_y.subtract(TILE_SIZE/2), nearest_pixel_intersection_x.add(TILE_SIZE/2), nearest_pixel_intersection_y.add(TILE_SIZE/2)], proj=self.proj, geodesic=False)
        self.crs = self.proj.getInfo()['crs'] # CRS of B4 band
        s2_image = s2_image.select([band for band in bands if band not in ['SCL', 'QA60']]).resample('bilinear').reproject(self.proj).addBands(s2_image.select(['SCL', 'QA60']).reproject(self.proj)) # bands with continuous pixel values get bilinear projection and those with categorical pixel values get nearest neighbor projection
        self.pixel_level_data['sentinel2'] = s2_image.rename([f'Sentinel2_{band}' if band not in ['MSK_CLDPRB', 'S2CLOUDLESS', 'SCL', 'QA60'] else band for band in bands])

    def sentinel1(self):
        bands = ['VV', 'VH', 'HH', 'HV']
        sentinel1_images = (ee.ImageCollection('COPERNICUS/S1_GRD')
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.tile) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.tile.buffer(200))) # gets images containing the tile plus some buffer
                              .filterMetadata('instrumentMode', 'equals', 'IW') # selects for the interferometric wide swath mode
                              .map(lambda image: image.set('date_difference', image.date().difference(self.s2_date, 'day').abs())) # calculate days off from S2 image
                              .sort('date_difference')) # sort in ascending order by days off

        asc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'ASCENDING').first() # ascending image
        desc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING').first() # descending image
        s1_image = None
        nan_band = ee.Image.constant(-9999).float().reproject(self.proj)

        # adding ascending bands
        for band in bands:
            band_data = nan_band.rename(f'Sentinel1_ascending_{band}')

            if asc_image.getInfo() is not None:
                if band in asc_image.bandNames().getInfo():
                    band_data = asc_image.select(band).float().resample('bilinear').reproject(self.proj).rename(f'Sentinel1_ascending_{band}')

            s1_image = band_data if s1_image is None else ee.Image.cat([s1_image, band_data])

        # adding descending bands
        for band in bands:
            band_data = nan_band.rename(f'Sentinel1_descending_{band}')

            if desc_image.getInfo() is not None:
                if band in desc_image.bandNames().getInfo():
                    band_data = desc_image.select(band).float().resample('bilinear').reproject(self.proj).rename(f'Sentinel1_descending_{band}')

            s1_image = ee.Image.cat([s1_image, band_data])

        self.pixel_level_data['sentinel1'] = s1_image

        if asc_image.getInfo() is None and desc_image.getInfo() is None: # if there is no ascending image and no descending image
            return False

    def aster(self):
        elevation = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').select('b1').float() # get elevation band
        slope = ee.Terrain.slope(elevation) # calculate slope from elevation data
        self.pixel_level_data['aster'] = ee.Image.cat([elevation, slope]).resample('bilinear').reproject(self.proj).rename(['AsterDEM_elevation', 'AsterDEM_slope']) # combine the elevation and slope into a single image

    def dynamic_world(self):
        '''
        This function gets the dynamic world data for the tile. The dynamic world data is a collection of images with the same name as the sentinel 2 image for that tile. It consist of 9 classes, we add one more to indicate missing
        information. The classes are as follows:
        0: No data
        1: Water
        2: Trees
        3: Grass
        4: Flooded vegetation
        5: Crops
        6: Shrub and scrub
        7: Built
        8: Bare
        9: Snow and ice

        We choose the label band since that contains which of these labels were chosen.
        '''

        year = self.s2_date.split('-')[0]
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        dynamic_world_images = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                                  .filterDate(start_date, end_date)
                                  .filterBounds(self.tile)
                                  .select('label'))

        def reclasify(image):
            label = image.select('label')
            label2 = label\
                    .where(image.eq(0), 1)\
                    .where(image.eq(1), 2)\
                    .where(image.eq(2), 3)\
                    .where(image.eq(3), 4)\
                    .where(image.eq(4), 5)\
                    .where(image.eq(5), 6)\
                    .where(image.eq(6), 7)\
                    .where(image.eq(7), 8)\
                    .where(image.eq(8), 9)\
                    .where(image.eq(9), 10)

            # replacing the label band with the new label band
            image = image.addBands(label2.rename('label2'))
            image = image.select('label2')
            image = image.rename('label')
            return image

        dynamic_world_images = dynamic_world_images.map(reclasify)
        dw_image = dynamic_world_images.mode()
        bands = dw_image.bandNames().getInfo()

        if len(bands) == 0:
            self.pixel_level_data['dynamic_world'] = ee.Image.constant(0).reproject(self.proj).rename('DynamicWorld')

            return False
        else:
            self.pixel_level_data['dynamic_world'] = dw_image.reproject(self.proj).rename('DynamicWorld')

    def canopy_height_eth(self):
        '''
        Gets the ETH canopy height and standard deviation from the year 2020
        '''

        height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').float()
        std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').float()
        self.pixel_level_data['canopy_height_eth'] = ee.Image.cat([height, std]).resample('bilinear').reproject(self.proj).rename(['ETHGCH_canopy_height', 'ETHGCH_canopy_height_uncertainty'])

    def esa_worldcover(self):
        ''' Gets the ESA worldcover data '''

        self.pixel_level_data['esa_worldcover'] = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').reproject(self.proj).rename('ESA_Worldcover')

    def era5(self):
        collection = 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
        bands = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum']
        year, month, _ = list(map(int, self.s2_date.split('-')))
        month_first_day = datetime(year, month, 1)
        month_last_day = (month_first_day + relativedelta(months=1, days=-1)).strftime('%Y-%m-%d')
        last_month = month - 1 if month > 1 else 12
        last_month_first_day = datetime(year, last_month, 1)
        last_month_last_day = (last_month_first_day + relativedelta(months=1, days=-1)).strftime('%Y-%m-%d')
        era5_month = (ee.ImageCollection(collection)
                          .filterDate(month_first_day, month_last_day)
                          .select(bands)
                          .toBands()
                          .rename(['temperature_month_mean', 'temperature_month_min', 'temperature_month_max', 'precipitation_month']))
        era5_last_month = (ee.ImageCollection(collection)
                          .filterDate(last_month_first_day, last_month_last_day)
                          .select(bands)
                          .toBands()
                          .rename(['temperature_last_month_mean', 'temperature_last_month_min', 'temperature_last_month_max', 'precipitation_last_month']))
        last_year_month_first_day = datetime(year-1, month, 1).strftime('%Y-%m-%d')
        era5_year = (ee.ImageCollection(collection)
                         .filterDate(last_year_month_first_day, month_last_day))
        band_reducer = {'temperature_2m':  ee.Reducer.mean(),
                        'temperature_2m_min': ee.Reducer.min(),
                        'temperature_2m_max': ee.Reducer.max(),
                        'total_precipitation_sum': ee.Reducer.sum()}
        era5_year = ee.ImageCollection([era5_year.select(band).reduce(band_reducer[band]) for band in bands]).toBands().rename(['temperature_year_mean', 'temperature_year_min', 'temperature_year_max', 'precipitation_year']).float()
        era5 = ee.Image.cat([era5_month, era5_last_month, era5_year]).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=10).getInfo()
        self.era5_data = era5

        if all(item is None for item in list(era5.values())):
            return False

    def task_data(self, task):
        if task == 'biomass':
            self.pixel_level_data['biomass'] = ee.Image.constant(-9999).paint(self.gedi_points, 'agbd').reproject(self.proj).rename('biomass') # image array with -9999 wherever there are no points, aligned with Sentinel-2

        elif task == 'species':
            month = self.point['properties']['month']

            if month > 1 and month < 12:
                allowed_months = [month-1, month, month+1]
            elif month == 1:
                allowed_months = [12, month, 2]
            elif month == 12:
                allowed_months = [11, month, 1]

            observations_selected_species_gdf = gpd.read_file('species/observations_selected_species_gdf.geojson')
            observations_in_tile = observations_selected_species_gdf[observations_selected_species_gdf.within(Polygon(self.tile.transform('EPSG:4326', maxError=1).getInfo()['coordinates'][0]))] # gets observations in the tile
            observations_in_tile_filtered = observations_in_tile[observations_in_tile['month'].isin(allowed_months)] # keeps observations within one month of the main species'
            species = list(observations_in_tile_filtered['species'].unique()) # all relevant species to the tile
            main_species = self.point['properties']['main_species']
            species_labels = utils.read_json('species/species_labels.json')
            species_numbers = [species_labels[species_] for species_ in species]
            main_species_number = species_labels[main_species]

            self.task_values = {'species': ','.join(map(str, species_numbers)),
                                'species_main': main_species_number,
                                'name_species': ','.join(species),
                                'name_species_main': main_species}

        elif 'soil' in task:
            self.task_values = {task: self.point['properties']['value']}

    def save_tiff(self, image):
        band_names = image.bandNames().getInfo()
        url = image.getDownloadUrl({'name': f'tile_{self.id}',
                                    'scale': 10,
                                    'crs': self.crs,
                                    'region': self.tile,
                                    'format': 'GeoTIFF',
                                    'bands': band_names})
        success = False
        tiff_path = f'{self.data_dir_path}/{self.task}/data/tile_{self.id}_data.tif'

        while not success:
            response = requests.get(url, stream=True, verify=certifi.where())

            if response.status_code != 200:
                continue

            with open(tiff_path, 'wb') as tiff: # writes in binary mode
                tiff.write(response.content)

            with rasterio.open(tiff_path) as tiff:
                array = tiff.read()
                assert array.shape[1] == TILE_SIZE
                assert array.shape[2] == TILE_SIZE

                # update the tags to include the image level and task data
                tags = tiff.tags()
                image_level_modalities = {'climate_temperature_month_mean': self.era5_data['temperature_month_mean'],
                                          'climate_temperature_last_month_mean': self.era5_data['temperature_last_month_mean'],
                                          'climate_temperature_year_mean': self.era5_data['temperature_year_mean'],
                                          'climate_temperature_month_max': self.era5_data['temperature_month_max'],
                                          'climate_temperature_last_month_max': self.era5_data['temperature_last_month_max'],
                                          'climate_temperature_year_max': self.era5_data['temperature_year_max'],
                                          'climate_temperature_month_min': self.era5_data['temperature_month_min'],
                                          'climate_temperature_last_month_min': self.era5_data['temperature_last_month_min'],
                                          'climate_temperature_year_min': self.era5_data['temperature_year_min'],
                                          'climate_precipitation_month': self.era5_data['precipitation_month'],
                                          'climate_precipitation_last_month': self.era5_data['precipitation_last_month'],
                                          'climate_precipitation_year': self.era5_data['precipitation_year'],
                                          'latitude_sin': self.geolocation_encoding['lat_sin'],
                                          'latitude_cos': self.geolocation_encoding['lat_cos'],
                                          'longitude_sin': self.geolocation_encoding['lon_sin'],
                                          'longitude_cos': self.geolocation_encoding['lon_cos'],
                                          'month_sin': self.month_encoding['month_sin'],
                                          'month_cos': self.month_encoding['month_cos'],
                                          'biome': self.biome,
                                          'ecoregion': self.ecoregion,
                                          'name_biome': self.name_biome,
                                          'name_ecoregion': self.name_ecoregion,
                                          'crs': self.crs,
                                          'lat': self.lat,
                                          'lon': self.lon,
                                          's2_date': self.s2_date,
                                          'missing_modalities': ','.join(self.missing_modalities),
                                          'MSK_CLDPRB_CLOUDY_PIXEL_FRACTION': self.pixel_level_data['sentinel2'].get('MSK_CLDPRB_CLOUDY_PIXEL_FRACTION').getInfo(),
                                          'S2CLOUDLESS_CLOUDY_PIXEL_FRACTION': self.pixel_level_data['sentinel2'].get('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION').getInfo()}

                if self.task_values:
                    tags.update(image_level_modalities | self.task_values)
                else:
                    tags.update(image_level_modalities)

                with rasterio.open(tiff_path, 'w', **tiff.meta) as updated_tiff:
                    updated_tiff.write(array)
                    updated_tiff.update_tags(**tags)

                    for i in range(tiff.count):
                        updated_tiff.update_tags(i+1, BAND_NAME=band_names[i])

            success = True
