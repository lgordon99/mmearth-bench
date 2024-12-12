'''
ee_data.py
A general class to collect the data from GEE. Each function in the class will be a different dataset, and they share common variables like the start and end date, the projection etc.
'''

# imports
from datetime import datetime
from dateutil.relativedelta import relativedelta
import certifi
import ee
import geopandas as gpd
import numpy as np
import rasterio
import requests
import time
import utils

TILE_SIZE_M = utils.read_yaml('config.yml')['TILE_SIZE_M']

class EEData:
    def __init__(self, point, task, start_end_dates):
        start = time.time()
        self.point = point
        self.bands = []
        self.crs = ''
        self.date_filter = ee.Filter.Or(*[ee.Filter.date(date_range[0], date_range[1]) for date_range in start_end_dates])
        self.s2_date = ''
        self.month_encoding = ''
        self.id = point['id']
        self.tile = ee.Geometry.Point(point['geometry']['coordinates']).buffer(TILE_SIZE_M / 2).bounds()
        self.lon, self.lat = point['geometry']['coordinates']
        self.geolocation_encoding = {'lat_sin': np.sin(np.deg2rad(self.lat)),
                                     'lat_cos': np.cos(np.deg2rad(self.lat)),
                                     'lon_sin': np.sin(np.deg2rad(self.lon)),
                                     'lon_cos': np.cos(np.deg2rad(self.lon))}
        self.task = task
        self.pixel_level_data = {}
        self.no_data = False
        self.era5_data = {}
        self.proj = ''
        self.task_values = None
        self.missing_modalities = []

        if 'biome' in list(point['properties'].keys()):
            self.biome_name = point['properties']['biome']
            self.ecoregion_name = point['properties']['ecoregion']
        else:
            if len(ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.tile).getInfo()['features']) > 0:
                self.biome_name = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.tile).getInfo()['features'][0]['properties']['BIOME_NAME']
                self.ecoregion_name = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.tile).getInfo()['features'][0]['properties']['ECO_NAME']
            else:
                self.biome_name = None
                self.ecoregion_name = None
                self.missing_modalities.append('biome/ecoregion')

        if self.biome_name is not None:
            biome_labels = utils.read_json('biomes_ecoregions_data/biome_labels.json')
            ecoregion_labels = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')

            self.biome = biome_labels[self.biome_name]
            self.ecoregion = ecoregion_labels[self.ecoregion_name]
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

    def sentinel2(self):
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B11', 'B12', 'SCL', 'MSK_CLDPRB', 'QA60']
        sentinel2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') # L2A collection
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.tile) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.tile.buffer(200)))) # gets images containing the tile plus some buffer
                            #   .map(lambda image: image.clip(self.tile))) # crops to tile
                            #   .map(lambda image: image.clip(self.buffered_polygon))) # crops to tile

        if sentinel2_images.size().getInfo() == 0: # if no images were returned
            return False

        msk_cldprob_res = sentinel2_images.select('MSK_CLDPRB').first().projection().nominalScale().getInfo() # resolution of MSK_CLDPRB band
        sentinel2_images = sentinel2_images.map(lambda image: image.set('cloudy_pixel_frac', image.select('MSK_CLDPRB').gte(0.1).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=msk_cldprob_res).get('MSK_CLDPRB'))).filter(ee.Filter.notNull(['cloudy_pixel_frac'])) # pixels >= 0.1 cloud probability are cloudy    
        cloudy_pixel_fracs = (sentinel2_images.aggregate_array('cloudy_pixel_frac').getInfo()) # gets the cloudy pixel fraction for each image
        least_cloudy_image_index = int(np.argmin(cloudy_pixel_fracs)) # gets the index for an image with the fewest cloudy pixels

        if cloudy_pixel_fracs[least_cloudy_image_index] >= 0.1: # if the least cloudy image has > 10% cloudy pixels
            return False # skipping tile

        s2_image = ee.Image(sentinel2_images.toList(sentinel2_images.size()).get(least_cloudy_image_index)).float() # get the least cloudy S2 image
        self.s2_date = s2_image.date().format('YYYY-MM-dd').getInfo() # date of S2 image
        month = int(self.s2_date.split('-')[1])
        self.month_encoding = {'month_sin': np.sin(np.pi * month / 6), 'month_cos': np.cos(np.pi * month / 6)}
        self.proj = s2_image.select('B4').projection() # projection of B4 band
        self.crs = self.proj.getInfo()['crs'] # CRS of B4 band
        s2_image = s2_image.select([band for band in bands if band not in ['SCL', 'QA60']]).resample('bilinear').reproject(self.proj).addBands(s2_image.select(['SCL', 'QA60']).reproject(self.proj))
        self.pixel_level_data['sentinel2'] = s2_image.rename([f'Sentinel2_{band}' if band not in ['SCL', 'MSK_CLDPRB', 'QA60'] else band for band in bands])

    def sentinel1(self):
        bands = ['VV', 'VH', 'HH', 'HV']
        sentinel1_images = (ee.ImageCollection('COPERNICUS/S1_GRD')
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.tile) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.tile.buffer(200))) # gets images containing the tile plus some buffer
                            #   .map(lambda image: image.clip(self.tile)) # crops to tile
                              .filterMetadata('instrumentMode', 'equals', 'IW') # selects for the interferometric wide swath mode
                              .map(lambda image: image.set('date_difference', image.date().difference(self.s2_date, 'day').abs())) # calculate days off from S2 image
                              .sort('date_difference')) # sort in ascending order by days off

        asc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'ASCENDING').first() # ascending image
        desc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING').first() # descending image
        s1_image = None
        # nan_band = ee.Image.constant(-9999).clip(self.tile).float().reproject(self.proj)
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
        # elevation = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').clip(self.tile).select('b1').float() # get elevation band
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
        # dw_image = dynamic_world_images.mode().clip(self.tile)
        dw_image = dynamic_world_images.mode()
        bands = dw_image.bandNames().getInfo()

        if len(bands) == 0:
            self.pixel_level_data['dynamic_world'] = None
        else:
            dw_image = dw_image.reproject(self.proj)
            self.pixel_level_data['dynamic_world'] = dw_image.rename('DynamicWorld')

    def canopy_height_eth(self):
        '''
        Gets the ETH canopy height and standard deviation from the year 2020
        '''

        # height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(self.tile).float()
        height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').float()
        # std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').clip(self.tile).float()
        std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').float()
        self.pixel_level_data['canopy_height_eth'] = ee.Image.cat([height, std]).resample('bilinear').reproject(self.proj).rename(['ETHGCH_canopy_height', 'ETHGCH_canopy_height_uncertainty'])

    def esa_worldcover(self):
        ''' Gets the ESA worldcover data '''

        # self.pixel_level_data['esa_worldcover'] = ee.ImageCollection('ESA/WorldCover/v100').first().clip(self.tile).select('Map').reproject(self.proj).rename('ESA_Worldcover')
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
                        #   .map(lambda image: image.clip(self.tile))
                          .select(bands)
                          .toBands()
                          .rename(['temperature_month_mean', 'temperature_month_min', 'temperature_month_max', 'precipitation_month']))
        era5_last_month = (ee.ImageCollection(collection)
                          .filterDate(last_month_first_day, last_month_last_day)
                        #   .map(lambda image: image.clip(self.tile))
                          .select(bands)
                          .toBands()
                          .rename(['temperature_last_month_mean', 'temperature_last_month_min', 'temperature_last_month_max', 'precipitation_last_month']))
        last_year_month_first_day = datetime(year-1, month, 1).strftime('%Y-%m-%d')
        era5_year = (ee.ImageCollection(collection)
                         .filterDate(last_year_month_first_day, month_last_day))
                        #  .map(lambda image: image.clip(self.tile)))
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
            collection_names = (ee.FeatureCollection('LARSE/GEDI/GEDI04_A_002_INDEX') # table index
                                .filter('time_start >= "2020-01-01" && time_end <= "2020-12-31"') # get feature collections with features in the selected year
                                .filterBounds(self.tile) # get feature collections that have features within the tile
                                .aggregate_array('table_id') # extract the IDs of the feature collections
                                .getInfo()) # list of names of the feature collections
            quality_filter = 'degrade_flag == 0 && l2_quality_flag == 1 && l4_quality_flag == 1 && leaf_off_flag == 0 && region_class > 0'
            gedi_points = (ee.FeatureCollection([result for _, result in ((collection_name, utils.get_asset_if_valid(ee.FeatureCollection(collection_name))) for collection_name in collection_names) if result is not None])
                            .flatten() # merge all the feature collections into one
                            .filterBounds(self.tile) # collection of the features that are within the tile
                            .filter(quality_filter) # apply the quality filter
                            .map(lambda point: point.set('off_after_on', ee.Number(point.get('leaf_off_doy')).subtract(ee.Number(point.get('leaf_on_doy'))))) # adding property for difference between leaf off and on days
                            .filter(ee.Filter.gt('off_after_on', 0))) # filtering by GEDI points with leaf on before leaf off
            self.pixel_level_data['biomass'] = ee.Image.constant(-9999).paint(gedi_points, 'agbd').reproject(self.proj).rename('biomass') # image array with -9999 wherever there are no points, aligned with Sentinel-2

        elif task == 'species':
            month = self.point['properties']['month']

            if month > 1 and month < 12:
                allowed_months = [month-1, month, month+1]
            elif month == 1:
                allowed_months = [12, month, 2]
            elif month == 12:
                allowed_months = [11, month, 1]

            observations_selected_species_gdf = gpd.read_file('species/observations_selected_species_gdf.geojson')
            observations_in_tile = observations_selected_species_gdf[observations_selected_species_gdf.within(self.tile)] # gets observations in the tile
            observations_in_tile_filtered = observations_in_tile[observations_in_tile['month'].isin(allowed_months)] # keeps observations within one month of the main species'
            species = list(observations_in_tile_filtered['species'].unique()) # all relevant species to the tile
            main_species = self.point['properties']['main_species']
            species_labels = utils.read_json('species/species_labels.json')
            main_species_number = species_labels[main_species]
            species_numbers = [species_labels[species_] for species_ in species]
            species_string = ''

            for i in range(len(species)):
                species_string += f'{species_numbers[i]},{species[i]}'

            self.task_values = {'species': species_string, 'main_species': f'{main_species_number},{main_species}'}

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
        tiff_path = f'{self.task}/data/tile_{self.id}_data.tif'

        while not success:
            response = requests.get(url, stream=True, verify=certifi.where())

            if response.status_code != 200:
                continue

            with open(tiff_path, 'wb') as tiff: # writes in binary mode
                tiff.write(response.content)

            with rasterio.open(tiff_path) as tiff:
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
                                          'biome': f'{self.biome},{self.biome_name}',
                                          'ecoregion': f'{self.ecoregion},{self.ecoregion_name}',
                                          'crs': self.crs,
                                          'lat': self.lat,
                                          'lon': self.lon,
                                          's2_date': self.s2_date,
                                          'missing_modalities': ','.join(self.missing_modalities)}

                if self.task_values:
                    tags.update(image_level_modalities | self.task_values)
                else:
                    tags.update(image_level_modalities)

                with rasterio.open(tiff_path, 'w', **tiff.meta) as updated_tiff:
                    updated_tiff.write(tiff.read())
                    updated_tiff.update_tags(**tags)

                    for i in range(tiff.count):
                        updated_tiff.update_tags(i+1, BAND_NAME=band_names[i])

            success = True
