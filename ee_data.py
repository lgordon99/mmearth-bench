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
import requests
import shutil
import time
import zipfile

class EEData:
    def __init__(self, tile, task, start_end_dates, biomass_points):
        self.tile = tile
        self.bands = []
        self.crs = ''
        self.start_end_dates = start_end_dates
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
        self.biome = tile['properties']['biome']
        self.ecoregion = tile['properties']['ecoregion']
        self.task = task
        self.image_set = {}
        self.no_data = False
        self.era5_data = {}
        self.proj = ''
        self.biomass_points = biomass_points
        start = time.time()
        datasets = ['sentinel2', 'sentinel1', 'aster', 'canopy_height_eth', 'dynamic_world', 'esa_worldcover', 'era5', 'gedi_agb']

        for function_name in datasets: # series of function calls to get the data
            if getattr(self, function_name)() is False: # if the method returns False
                print(f'{function_name} returned False')
                print(f'Skipping tile with ID {self.id}')
                self.no_data = True
                break

        if not self.no_data:
            merged_image = self.image_set[datasets[0]] # start with Sentinel-2

            for dataset, value in self.image_set.items():
                if dataset == datasets[0]:
                    continue # already included Sentinel-2
                else:
                    merged_image = ee.Image.cat([merged_image, value])

            self.export_local_single(merged_image)

            print(f'Time elapsed for this tile: {round(time.time() - start, 2)}s')

    def sentinel2(self):
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B9', 'B11', 'B12', 'SCL', 'MSK_CLDPRB', 'QA60']
        sentinel2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') # L2A collection
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.polygon) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.polygon.buffer(200))) # gets images containing the tile plus some buffer
                              .map(lambda image: image.clip(self.polygon))) # crops to tile

        print(f'{sentinel2_images.size().getInfo()} Sentinel-2 images returned')

        msk_cldprob_res = sentinel2_images.select('MSK_CLDPRB').first().projection().nominalScale().getInfo() # resolution of MSK_CLDPRB band
        cloudy_pixel_frac_list = ((sentinel2_images
                                   .select('MSK_CLDPRB') # select MSK_CLDPRB band
                                   .map(lambda image: image.set('cloudy_pixel_frac', image.gte(0.1).reduceRegion(reducer=ee.Reducer.mean(), scale=msk_cldprob_res)))) # pixels >= 0.1 cloud probability are cloudy
                                  .aggregate_array('cloudy_pixel_frac').getInfo()) # get the cloudy pixel fraction for each image
        cloudy_pixel_fracs = [list(item.values())[0] for item in cloudy_pixel_frac_list] # extract just the cloudy pixel fraction value
        least_cloudy_image_index = int(np.argmin(cloudy_pixel_fracs)) # get the index for an image with the fewest cloudy pixels

        if cloudy_pixel_fracs[least_cloudy_image_index] >= 0.1: # if the least cloudy image has > 10% cloudy pixels
            return False # skipping tile

        s2_image = ee.Image(sentinel2_images.toList(sentinel2_images.size()).get(least_cloudy_image_index)).float() # get the least cloudy S2 image
        self.s2_date = s2_image.date().format('YYYY-MM-dd').getInfo() # date of S2 image
        month = int(self.s2_date.split('-')[1])
        self.month_encoding = {'sin': np.sin(np.pi * month / 6), 'cos': np.cos(np.pi * month / 6)}
        self.proj = s2_image.select('B4').projection() # projection of B4 band
        self.crs = self.proj.getInfo()['crs'] # CRS of B4 band
        self.image_set['sentinel2'] = s2_image.select([band for band in bands if band not in ['SCL', 'QA60']]).resample('bilinear').reproject(self.proj).addBands(s2_image.select(['SCL', 'QA60']).reproject(self.proj))

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
                band_data = asc_image.select(band).float().resample('bilinear').reproject(self.proj).rename(f'asc_{band}')
            else:
                band_data = nan_band.rename(f'asc_{band}')

            s1_image = band_data if s1_image is None else ee.Image.cat([s1_image, band_data])

        # adding descending bands
        for band in bands:
            if band in desc_image.bandNames().getInfo():
                band_data = desc_image.select(band).float().resample('bilinear').reproject(self.proj).rename(f'desc_{band}')
            else:
                band_data = nan_band.rename(f'desc_{band}')

            s1_image = band_data if s1_image is None else ee.Image.cat([s1_image, band_data])

        self.image_set['sentinel1'] = s1_image

    def aster(self):
        elevation = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').clip(self.polygon).select('b1').float() # get elevation band
        slope = ee.Terrain.slope(elevation) # calculate slope from elevation data
        self.image_set['aster'] = ee.Image.cat([elevation, slope]).resample('bilinear').reproject(self.proj) # combine the elevation and slope into a single image

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

        # cfg = self.cfg['dynamic_world']
        # data_name = cfg['name']
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
        # print(dw_image.sampleRectangle(region=self.polygon).get('label2').getInfo())
        bands = dw_image.bandNames().getInfo()

        if len(bands) == 0:
            self.image_set['dynamic_world'] = None
        else:
            dw_image = dw_image.reproject(self.proj)
            self.image_set['dynamic_world'] = dw_image

    def canopy_height_eth(self):
        '''
        Gets the ETH canopy height and standard deviation from the year 2020
        '''

        height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(self.polygon).float()
        std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').clip(self.polygon).float()
        self.image_set['canopy_height_eth'] = ee.Image.cat([height, std]).resample('bilinear').reproject(self.proj).rename(['height', 'std'])

    def esa_worldcover(self):
        ''' Gets the ESA worldcover data '''

        self.image_set['esa_worldcover'] = ee.ImageCollection('ESA/WorldCover/v100').first().clip(self.polygon).select('Map').reproject(self.proj)

    def era5(self):
        collection = 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
        bands = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum']
        year, month, _ = list(map(int, self.s2_date.split('-')))
        last_day_of_current_month = ((datetime(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
        first_day_current_month_prev_year = datetime(year-1, month, 1).strftime('%Y-%m-%d')
        first_day_prev_month = f'{year}-{month-1}-01' if month > 1 else f'{year-1}-12-01'
        era5_monthly = (ee.ImageCollection(collection)
                          .filterDate(first_day_prev_month, last_day_of_current_month) # previous month and current month
                          .map(lambda image: image.clip(self.polygon))
                          .select(bands)
                          .toBands()
                          .rename([f'this_month_{band}' if i < 4 else f'last_month_{band}' for i, band in enumerate(2 * bands)]))
        era5_yearly = (ee.ImageCollection(collection)
                         .filterDate(first_day_current_month_prev_year, last_day_of_current_month)
                         .map(lambda image: image.clip(self.polygon)))

        def compute_yearly_stat(band):
            if band == 'temperature_2m':
                reducer = ee.Reducer.mean()
            elif band == 'temperature_2m_min':
                reducer = ee.Reducer.min()
            elif band == 'temperature_2m_max':
                reducer = ee.Reducer.max()
            elif band == 'total_precipitation_sum':
                reducer = ee.Reducer.sum()

            return era5_yearly.select(band).reduce(reducer)

        era5_yearly_image = ee.ImageCollection([compute_yearly_stat(band) for band in bands]).toBands().rename([f'yearly_{band}' for band in bands]).float()
        era5_image = ee.Image.cat([era5_monthly, era5_yearly_image])
        era5_image_bands = era5_image.bandNames().getInfo()
        era5_image = era5_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=self.polygon, scale=10).getInfo()

        if all(item is None for item in list(era5_image.values())):
            return False

        self.era5_data = {band: era5_image[band] for band in era5_image_bands}

    def gedi_agb(self):
        '''Gets GEDI aboveground biomass data'''

        image = ee.Image.constant(-9999).paint(self.biomass_points, 'agbd').reproject(self.proj) # image array with -9999 wherever there are no points, aligned with Sentinel-2
        self.image_set['gedi_agb'] = image

    def export_local_single(self, image):
        url = image.getDownloadUrl({'name': f'tile_{self.id}',
                                    'scale': 10,
                                    'crs': self.crs,
                                    'region': self.polygon.getInfo()['coordinates'],
                                    'format': 'GeoTIFF',
                                    'bands': image.bandNames().getInfo()})
        response = requests.get(url, stream=True, verify=certifi.where())

        with open(f'tiles/{self.task}/pixel_level_data/tile_{self.id}_pixel_level_data.tif', 'wb') as f:
            f.write(response.content)
