'''
ee_data.py
A general class to collect the data from GEE. Each function in the class will be a different dataset, and they share common variables like the start and end date, the projection etc.
'''

# imports
from datetime import datetime, timedelta
from utils import read_json
import certifi
import ee
import hashlib
import logging
import math
import numpy as np
import os
import pandas as pd
import rasterio
import requests
import shutil
import time
import zipfile

class EEData:
    def __init__(self, tile, task, start_end_dates, task_values):
        start = time.time()

        self.tile = tile
        self.bands = []
        self.crs = ''
        self.date_filter = ee.Filter.Or(*[ee.Filter.date(date_range[0], date_range[1]) for date_range in start_end_dates])
        self.s2_date = ''
        self.month_encoding = ''
        self.id = tile['id']
        self.polygon = ee.Geometry.Polygon(tile['geometry']['coordinates'])
        self.lon = self.polygon.centroid().coordinates().get(0).getInfo()
        self.lat = self.polygon.centroid().coordinates().get(1).getInfo()
        self.geolocation_encoding = {'lat_sin': np.sin(np.deg2rad(self.lat)),
                                     'lat_cos': np.cos(np.deg2rad(self.lat)),
                                     'lon_sin': np.sin(np.deg2rad(self.lon)),
                                     'lon_cos': np.cos(np.deg2rad(self.lon))}
        self.task = task
        self.modality_data = {}
        self.no_data = False
        self.era5_data = {}
        self.proj = ''
        self.task_values = task_values
        self.modality_returned_false = ''

        if 'biome' in list(tile['properties'].keys()):
            self.biome = tile['properties']['biome']
            self.ecoregion = tile['properties']['ecoregion']
        else:
            if len(ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.polygon).getInfo()['features']) > 0:
                self.biome = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.polygon).getInfo()['features'][0]['properties']['BIOME_NAME']
                self.ecoregion = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.polygon).getInfo()['features'][0]['properties']['ECO_NAME']
            else:
                self.biome = None
                self.ecoregion = None

        datasets = ['sentinel2', 'sentinel1', 'aster', 'canopy_height_eth', 'dynamic_world', 'esa_worldcover', 'era5']

        for function_name in datasets: # series of function calls to get the data
            if getattr(self, function_name)() is False: # if the method returns False
                self.no_data = True
                self.modality_returned_false = function_name
                print(f'{function_name} returned False')

                break

        if task == 'biomass':
            self.biomass()

        if not self.no_data:
            merged_image = self.modality_data[datasets[0]] # start with Sentinel-2

            for dataset, value in self.modality_data.items():
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
                              .filterBounds(self.polygon) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.polygon.buffer(200))) # gets images containing the tile plus some buffer
                              .map(lambda image: image.clip(self.polygon))) # crops to tile

        if sentinel2_images.size().getInfo() == 0: # if no images were returned
            return False

        msk_cldprob_res = sentinel2_images.select('MSK_CLDPRB').first().projection().nominalScale().getInfo() # resolution of MSK_CLDPRB band
        sentinel2_images = sentinel2_images.map(lambda image: image.set('cloudy_pixel_frac', image.select('MSK_CLDPRB').gte(0.1).reduceRegion(reducer=ee.Reducer.mean(), scale=msk_cldprob_res).get('MSK_CLDPRB'))).filter(ee.Filter.notNull(['cloudy_pixel_frac'])) # pixels >= 0.1 cloud probability are cloudy    
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
        self.modality_data['sentinel2'] = s2_image.rename([f'Sentinel2_{band}' for band in bands])

    def sentinel1(self):
        bands = ['VV', 'VH', 'HH', 'HV']
        sentinel1_images = (ee.ImageCollection('COPERNICUS/S1_GRD')
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.polygon) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.polygon.buffer(200))) # gets images containing the tile plus some buffer
                              .map(lambda image: image.clip(self.polygon)) # crops to tile
                              .filterMetadata('instrumentMode', 'equals', 'IW') # selects for the interferometric wide swath mode
                              .map(lambda image: image.set('date_difference', image.date().difference(self.s2_date, 'day').abs())) # calculate days off from S2 image
                              .sort('date_difference')) # sort in ascending order by days off

        asc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'ASCENDING').first() # ascending image
        desc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING').first() # descending image

        if asc_image.getInfo() is None or desc_image.getInfo() is None: # if there is no ascending image or no descending image
            return False # skip tile

        s1_image = None
        nan_band = ee.Image.constant(-9999).clip(self.polygon).float().reproject(self.proj)

        # adding ascending bands
        for band in bands:
            if band in asc_image.bandNames().getInfo():
                band_data = asc_image.select(band).float().resample('bilinear').reproject(self.proj).rename(f'Sentinel1_ascending_{band}')
            else:
                band_data = nan_band.rename(f'Sentinel1_ascending_{band}')

            s1_image = band_data if s1_image is None else ee.Image.cat([s1_image, band_data])

        # adding descending bands
        for band in bands:
            if band in desc_image.bandNames().getInfo():
                band_data = desc_image.select(band).float().resample('bilinear').reproject(self.proj).rename(f'Sentinel1_descending_{band}')
            else:
                band_data = nan_band.rename(f'Sentinel1_descending_{band}')

            s1_image = band_data if s1_image is None else ee.Image.cat([s1_image, band_data])

        self.modality_data['sentinel1'] = s1_image

    def aster(self):
        elevation = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').clip(self.polygon).select('b1').float() # get elevation band
        slope = ee.Terrain.slope(elevation) # calculate slope from elevation data
        self.modality_data['aster'] = ee.Image.cat([elevation, slope]).resample('bilinear').reproject(self.proj).rename(['AsterDEM_elevation', 'AsterDEM_slope']) # combine the elevation and slope into a single image

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
                                  .filterBounds(self.polygon)
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
        dw_image = dynamic_world_images.mode().clip(self.polygon)
        bands = dw_image.bandNames().getInfo()

        if len(bands) == 0:
            self.modality_data['dynamic_world'] = None
        else:
            dw_image = dw_image.reproject(self.proj)
            self.modality_data['dynamic_world'] = dw_image.rename('DynamicWorld')

    def canopy_height_eth(self):
        '''
        Gets the ETH canopy height and standard deviation from the year 2020
        '''

        height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(self.polygon).float()
        std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').clip(self.polygon).float()
        self.modality_data['canopy_height_eth'] = ee.Image.cat([height, std]).resample('bilinear').reproject(self.proj).rename(['ETHGCH_canopy_height', 'ETHGCH_canopy_height_uncertainty'])

    def esa_worldcover(self):
        ''' Gets the ESA worldcover data '''

        self.modality_data['esa_worldcover'] = ee.ImageCollection('ESA/WorldCover/v100').first().clip(self.polygon).select('Map').reproject(self.proj).rename('ESA_Worldcover')

    def era5(self):
        from dateutil.relativedelta import relativedelta

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
                          .map(lambda image: image.clip(self.polygon))
                          .select(bands)
                          .toBands()
                          .rename(['temperature_month_mean', 'temperature_month_min', 'temperature_month_max', 'precipitation_month']))
        era5_last_month = (ee.ImageCollection(collection)
                          .filterDate(last_month_first_day, last_month_last_day)
                          .map(lambda image: image.clip(self.polygon))
                          .select(bands)
                          .toBands()
                          .rename(['temperature_last_month_mean', 'temperature_last_month_min', 'temperature_last_month_max', 'precipitation_last_month']))
        last_year_month_first_day = datetime(year-1, month, 1).strftime('%Y-%m-%d')
        era5_year = (ee.ImageCollection(collection)
                         .filterDate(last_year_month_first_day, month_last_day)
                         .map(lambda image: image.clip(self.polygon)))
        band_reducer = {'temperature_2m':  ee.Reducer.mean(),
                        'temperature_2m_min': ee.Reducer.min(),
                        'temperature_2m_max': ee.Reducer.max(),
                        'total_precipitation_sum': ee.Reducer.sum()}
        era5_year = ee.ImageCollection([era5_year.select(band).reduce(band_reducer[band]) for band in bands]).toBands().rename(['temperature_year_mean', 'temperature_year_min', 'temperature_year_max', 'precipitation_year']).float()
        era5 = ee.Image.cat([era5_month, era5_last_month, era5_year]).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.polygon, scale=10).getInfo()

        if all(item is None for item in list(era5.values())):
            return False

        self.era5_data = era5

    def biomass(self):
        '''Gets GEDI aboveground biomass data'''

        image = ee.Image.constant(-9999).paint(self.task_values, 'agbd').reproject(self.proj) # image array with -9999 wherever there are no points, aligned with Sentinel-2
        self.modality_data['biomass'] = image

    def save_tiff(self, image):
        band_names = image.bandNames().getInfo()
        url = image.getDownloadUrl({'name': f'tile_{self.id}',
                                    'scale': 10,
                                    'crs': self.crs,
                                    'region': self.polygon.getInfo()['coordinates'],
                                    'format': 'GeoTIFF',
                                    'bands': band_names})
        response = requests.get(url, stream=True, verify=certifi.where())
        tiff_path = f'{self.task}/data/tile_{self.id}_data.tif'

        with open(tiff_path, 'wb') as tiff: # writes in binary mode
            tiff.write(response.content)

        while not os.path.exists(tiff_path):
            time.sleep(1)

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
                                      'biome': self.biome,
                                      'ecoregion': self.ecoregion,
                                      'crs': self.crs,
                                      'lat': self.lat,
                                      'lon': self.lon,
                                      's2_date': self.s2_date}

            if isinstance(self.task_values, dict):
                tags.update(image_level_modalities | self.task_values)
            else:
                tags.update(image_level_modalities)

            with rasterio.open(tiff_path, 'w', **tiff.meta) as updated_tiff:
                updated_tiff.write(tiff.read())
                updated_tiff.update_tags(**tags)

                for i in range(tiff.count):
                    updated_tiff.update_tags(i+1, BAND_NAME=band_names[i])
