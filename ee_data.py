'''
ee_data.py
'''

# ============================================== IMPORTS ============================================== #

from datetime import datetime
from dateutil.relativedelta import relativedelta
from filelock import FileLock
from sys import argv
import certifi
import ee
import json
import numpy as np
import pandas as pd
import rasterio
import requests
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

TILE_SIZE_M = utils.read_yaml('config.yml')['TILE_SIZE_M']
resolution = 10 # meters per pixel
TILE_SIZE = int(TILE_SIZE_M / resolution)
OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']
year = '2020'
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
no_data_values = utils.read_json(f'{data_dir_path}/no_data_values.json')

# ============================================== FUNCTIONS ============================================== #

def get_last_day_of_month(month):
    return (datetime(int(year), month, 1) + relativedelta(months=1, days=-1)).day

# ============================================== CLASSES ============================================== #

class EEData:
    def __init__(self, task, point_id):
        start = time.time()

        self.task = task
        self.id = point_id
        self.point = utils.read_geojson(f'{data_dir_path}/{task}/{task}_points.geojson')['features'][point_id] # reading the GeoJSON file
        self.missing_modalities = []
        self.point_coordinates = self.point['geometry']['coordinates'] # [longitude, latitude]

        if task == 'biomass':
            self.gedi_points = utils.get_gedi_points(ee.Geometry.Point(self.point_coordinates).buffer(OUTER_TILE_SIZE_M / 2)) # GEDI points in the outer tile

        self.date_filter = ee.Filter.Or(*[ee.Filter.date(date_range[0], date_range[1]) for date_range in self.get_dates()])
        self.tile = ee.Geometry.Point(self.point_coordinates).buffer(TILE_SIZE_M / 2)
        self.pixel_level_data = {}
        self.tile_level_data = {}

        print(f'Running tile {self.id}')
        datasets = ['sentinel2', 'sentinel1', 'aster_gdem', 'eth_gch', 'dynamic_world', 'esa_worldcover', 'precipitation', 'temperature']

        for function_name in datasets: # series of function calls to get the data
            if getattr(self, function_name)() is False: # if the method returns False
                self.report_missing_modality(function_name) # saves the modality as missing

                if function_name == 'sentinel2':
                    break # does not collect data for the rest of the modalities

        if 'sentinel2' not in self.missing_modalities:
            ecoregion_features = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filterBounds(self.tile_center).getInfo()['features'] # data for the ecoregion containing the tile

            if len(ecoregion_features) > 0: # if the tile is in an ecoregion
                biome_name = ecoregion_features[0]['properties']['BIOME_NAME']
                ecoregion_name = ecoregion_features[0]['properties']['ECO_NAME']
            else:
                biome_name = 'N/A'
                ecoregion_name = 'N/A'

            if biome_name == 'N/A':
                biome = no_data_values['biome']
                ecoregion = no_data_values['ecoregion']

                self.missing_modalities.append('biome')
                self.missing_modalities.append('ecoregion')
            else:
                biome = utils.read_json('biomes_ecoregions_data/biome_labels.json')[biome_name]
                ecoregion = utils.read_json('biomes_ecoregions_data/ecoregion_labels.json')[ecoregion_name]

            self.tile_level_data['biome'] = biome
            self.tile_level_data['ecoregion'] = ecoregion
            self.tile_level_data['missing_modalities'] = json.dumps(self.missing_modalities) # saves the missing modalities as a JSON string
            self.task_data() # gets the task data for the tile

            merged_image = self.pixel_level_data[datasets[0]] # start with Sentinel-2

            for dataset, value in self.pixel_level_data.items():
                if dataset == datasets[0]:
                    continue # already included Sentinel-2
                else:
                    merged_image = ee.Image.cat([merged_image, value])

            self.save_tiff(merged_image)

        tile_ids_run_path = f'{data_dir_path}/{task}/{task}_tile_ids_run.json'
        lock = FileLock(f'{tile_ids_run_path}.lock')

        with lock:
            tile_ids_run = utils.read_json(tile_ids_run_path)
            tile_ids_run[str(self.id)] = 'missing' if 'sentinel2' in self.missing_modalities else 'done'

            with open(tile_ids_run_path, 'w') as file:
                json.dump(tile_ids_run, file, indent=4)

        print(f'Time elapsed for this tile: {round(time.time() - start, 2)}s')

    def get_dates(self):
        if self.task == 'biomass':
            leaf_on_off = np.array([self.gedi_points.aggregate_array('leaf_on_doy').getInfo(), self.gedi_points.aggregate_array('leaf_off_doy').getInfo()]).T # pairs of leaf on and off days for each point
            leaf_on_off_unique = list(map(list, set(map(tuple, leaf_on_off)))) # unique pairs
            leaf_on_off_dates = [[pd.to_datetime(pair[0], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d'), pd.to_datetime(pair[1], unit='D', origin=pd.Timestamp(f'{year}-01-01')).date().strftime('%Y-%m-%d')] for pair in leaf_on_off_unique] # converts the days to dates

            return leaf_on_off_dates
        else:
            point_latitude = self.point['geometry']['coordinates'][1]

            if point_latitude > 0: # if the point is in the northern hemisphere
                dates = [[f'{year}-{str(5).zfill(2)}-01', f'{year}-{str(9).zfill(2)}-{get_last_day_of_month(9)}']] # May - September
            elif point_latitude < 0: # if the point is in the southern hemisphere
                dates = [[f'{year}-{str(11).zfill(2)}-01', f'{year}-{str(12).zfill(2)}-{get_last_day_of_month(12)}'], [f'{year}-{str(1).zfill(2)}-01', f'{year}-{str(3).zfill(2)}-{get_last_day_of_month(3)}']] # November - March

            return dates

    def report_missing_modality(self, modality):
        self.missing_modalities.append(modality)

        print(f'{modality} is missing')

    def sentinel2(self):
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'MSK_CLDPRB', 'S2CLOUDLESS', 'SCL'] # bands to collect
        sentinel2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') # L2A collection
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.tile) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.tile.buffer(200)))) # gets images containing the tile plus some buffer

        if sentinel2_images.size().getInfo() == 0: # if no images passed the date and location filters
            return False

        msk_cldprob_res = sentinel2_images.select('MSK_CLDPRB').first().projection().nominalScale().getInfo() # resolution of MSK_CLDPRB band
        sentinel2_images = sentinel2_images.map(lambda image: image.set('MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', image.select('MSK_CLDPRB').gte(10).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=msk_cldprob_res).get('MSK_CLDPRB'))).filter(ee.Filter.notNull(['MSK_CLDPRB_CLOUDY_PIXEL_FRACTION'])) # pixels >= 10% cloud probability are cloudy
        sentinel2_images = sentinel2_images.filterMetadata('MSK_CLDPRB_CLOUDY_PIXEL_FRACTION', 'less_than', 0.1) # keeps images with less than 10% cloudy pixels

        if sentinel2_images.size().getInfo() == 0: # if no images passed the MSK_CLDPRB filter
            return False

        s2cloudless_images = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                                .filter(self.date_filter)
                                .filterBounds(self.tile)
                                .filter(ee.Filter.contains('.geo', self.tile.buffer(200))))
        sentinel2_images = ee.ImageCollection(ee.Join.saveFirst('S2CLOUDLESS').apply(**{'primary': sentinel2_images, 'secondary': s2cloudless_images, 'condition': ee.Filter.equals(**{'leftField': 'system:index', 'rightField': 'system:index'})})) # saves the s2cloudless images as properties of the corresponding Sentinel-2 images
        sentinel2_images = sentinel2_images.map(lambda image: image.addBands(ee.Image(image.get('S2CLOUDLESS')).rename('S2CLOUDLESS'))) # saves the s2cloudless images as a new band in the Sentinel-2 images called S2CLOUDLESS
        s2cloudless_res = s2cloudless_images.select('probability').first().projection().nominalScale().getInfo() # resolution of s2cloudless
        sentinel2_images = sentinel2_images.map(lambda image: image.set('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', image.select('S2CLOUDLESS').gte(10).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=s2cloudless_res).get('S2CLOUDLESS'))) # pixels >= 10% cloud probability are cloudy
        sentinel2_images = sentinel2_images.filterMetadata('S2CLOUDLESS_CLOUDY_PIXEL_FRACTION', 'less_than', 0.1) # keeps images with less than 10% cloudy pixels

        if sentinel2_images.size().getInfo() == 0: # if no images passed the S2CLOUDLESS filter
            return False

        scl_res = sentinel2_images.select('SCL').first().projection().nominalScale().getInfo() # resolution of SCL band
        sentinel2_images = sentinel2_images.map(lambda image: image.set('SCL_NO_DATA_PIXEL_FRACTION', image.select('SCL').unmask().eq(0).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=scl_res).get('SCL'))) # calculates the fraction of invalid pixels (SCL = 0)
        sentinel2_images = sentinel2_images.filterMetadata('SCL_NO_DATA_PIXEL_FRACTION', 'less_than', 0.1).sort('SCL_NO_DATA_PIXEL_FRACTION') # keeps images with less than 10% no data pixels and sorts them according to their fraction of no data pixels

        if sentinel2_images.size().getInfo() == 0: # if no images passed the SCL filter
            return False

        s2_image = sentinel2_images.first().float() # get the S2 image with the most valid pixels
        self.tile_level_data['sentinel2_system_index'] = s2_image.get('system:index').getInfo()
        self.tile_level_data['sentinel2_date'] = s2_image.date().format('YYYY-MM-dd').getInfo() # date of Sentinel-2 image
        month = int(self.tile_level_data['sentinel2_date'].split('-')[1]) # extracts the month of the Sentinel-2 image
        self.tile_level_data['month_encoding'] = json.dumps([np.cos(np.pi * month / 6), np.sin(np.pi * month / 6)]) # cyclic month encoding
        self.proj = s2_image.select('B4').projection() # projection of B4 band
        self.crs = self.proj.getInfo()['crs'] # CRS of B4 band
        projected_point_coordinates = ee.Geometry.Point(self.point_coordinates).transform(self.proj).coordinates() # projects the point onto the Sentinel-2 grid
        nearest_pixel_intersection_x = ee.Number(projected_point_coordinates.get(0)).round() # rounds the longitude to the nearest pixel intersection in the Sentinel-2 grid
        nearest_pixel_intersection_y = ee.Number(projected_point_coordinates.get(1)).round() # rounds the latitude to the nearest pixel intersection in the Sentinel-2 grid
        self.tile = ee.Geometry.Rectangle([nearest_pixel_intersection_x.subtract(TILE_SIZE/2), nearest_pixel_intersection_y.subtract(TILE_SIZE/2), nearest_pixel_intersection_x.add(TILE_SIZE/2), nearest_pixel_intersection_y.add(TILE_SIZE/2)], proj=self.proj, geodesic=False) # resets the tile to be centered at the nearest pixel intersection in the Sentinel-2 grid
        self.tile_center = self.tile.centroid(maxError=1)
        longitude, latitude = self.tile_center.coordinates().getInfo()
        self.tile_level_data['geolocation'] = json.dumps([longitude, latitude]) # gets the lon, lat coordinates of the tile centroid
        self.tile_level_data['geolocation_encoding'] = json.dumps([np.cos(np.deg2rad(longitude)), # cyclic location encding
                                                                   np.sin(np.deg2rad(longitude)),
                                                                   np.cos(np.deg2rad(latitude)),
                                                                   np.sin(np.deg2rad(latitude))])
        continuous_valued_bands = s2_image.select([band for band in bands if band != 'SCL']).resample('bilinear').reproject(self.proj).unmask(no_data_values['Sentinel2'])
        scl = s2_image.select('SCL').reproject(self.proj).unmask() # unmasks SCL to 0
        s2_image = continuous_valued_bands.addBands(scl) # combines continuous and categorical bands
        self.tile_level_data['MSK_CLDPRB_CLOUDY_PIXEL_FRACTION'] = s2_image.select('MSK_CLDPRB').gte(10).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=msk_cldprob_res).get('MSK_CLDPRB').getInfo()
        self.tile_level_data['S2CLOUDLESS_CLOUDY_PIXEL_FRACTION'] = s2_image.select('S2CLOUDLESS').gte(10).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=s2cloudless_res).get('S2CLOUDLESS').getInfo()
        self.tile_level_data['SCL_NO_DATA_PIXEL_FRACTION'] = s2_image.select('SCL').unmask().eq(0).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=scl_res).get('SCL').getInfo()
        self.pixel_level_data['sentinel2'] = s2_image.rename([f'Sentinel2_{band}' if band not in ['MSK_CLDPRB', 'S2CLOUDLESS', 'SCL'] else band for band in bands])

        sentinel2_image_main_bands = self.pixel_level_data['sentinel2'].select([band for band in self.pixel_level_data['sentinel2'].bandNames().getInfo() if 'Sentinel2' in band])
        band_nan_fractions = sentinel2_image_main_bands.eq(no_data_values['Sentinel2']).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=resolution).getInfo()

        if all(value == 1 for value in band_nan_fractions.values()): # if all Sentinel-2 pixels are no data
            return False

    def sentinel1(self):
        bands = ['VV', 'VH', 'HH', 'HV']
        sentinel1_images = (ee.ImageCollection('COPERNICUS/S1_GRD')
                              .filter(self.date_filter) # gets images in any of the specified date ranges
                              .filterBounds(self.tile) # gets images that have some overlap with the tile
                              .filter(ee.Filter.contains('.geo', self.tile.buffer(200))) # gets images containing the tile plus some buffer
                              .filterMetadata('instrumentMode', 'equals', 'IW') # selects for the interferometric wide swath mode
                              .map(lambda image: image.set('date_difference', image.date().difference(self.tile_level_data['sentinel2_date'], 'day').abs())) # calculate days off from S2 image
                              .sort('date_difference')) # sort in ascending order by days off
        asc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'ASCENDING').first() # ascending image
        desc_image = sentinel1_images.filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING').first() # descending image
        s1_image = None
        nan_band = ee.Image.constant(no_data_values['Sentinel1']).float().reproject(self.proj)

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

        self.pixel_level_data['sentinel1'] = s1_image.unmask(no_data_values['Sentinel1'])

        if asc_image.getInfo() is None and desc_image.getInfo() is None: # if there is no ascending image and no descending image
            return False

    def aster_gdem(self):
        elevation = ee.Image('projects/sat-io/open-datasets/ASTER/GDEM').select('b1').float() # elevation band
        elevation = elevation.mask(elevation.neq(no_data_values['ASTER_GDEM'])) # masks NaN values in the elevation band
        slope = ee.Terrain.slope(elevation) # calculates slope from elevation data
        self.pixel_level_data['aster_gdem'] = ee.Image.cat([elevation, slope]).resample('bilinear').reproject(self.proj).unmask(no_data_values['ASTER_GDEM']).rename(['ASTER_GDEM_elevation', 'ASTER_GDEM_slope']) # combines the elevation and slope into a single image

        if self.pixel_level_data['aster_gdem'].eq(no_data_values['ASTER_GDEM']).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=resolution).get('ASTER_GDEM_elevation').getInfo() == 1: # if all pixels in the tile are no data
            return False # if all pixels are no data, return False

    def eth_gch(self):
        '''
        Gets the ETH canopy height and standard deviation from the year 2020
        '''

        height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').float()
        std = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').float()
        self.pixel_level_data['eth_gch'] = ee.Image.cat([height, std]).resample('bilinear').reproject(self.proj).unmask(no_data_values['ETH_GCH']).rename(['ETH_GCH_height', 'ETH_GCH_uncertainty'])

        if self.pixel_level_data['eth_gch'].eq(no_data_values['ETH_GCH']).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=resolution).get('ETH_GCH_height').getInfo() == 1: # if all pixels in the tile are no data
            return False # if all pixels are no data, return False

    def dynamic_world(self):
        year = self.tile_level_data['sentinel2_date'].split('-')[0]
        dynamic_world_image = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                                  .filterBounds(self.tile)
                                  .select('label')
                                  .mode()) # gets the most common label for each pixel

        if len(dynamic_world_image.bandNames().getInfo()) == 0:
            self.pixel_level_data['dynamic_world'] = ee.Image.constant(no_data_values['DynamicWorld']).reproject(self.proj).rename('DynamicWorld')
        else:
            self.pixel_level_data['dynamic_world'] = dynamic_world_image.reproject(self.proj).unmask(no_data_values['DynamicWorld']).rename('DynamicWorld')

        if self.pixel_level_data['dynamic_world'].eq(no_data_values['DynamicWorld']).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=resolution).get('DynamicWorld').getInfo() == 1: # if all pixels in the tile are no data
            return False # if all pixels are no data, return False

    def esa_worldcover(self):
        self.pixel_level_data['esa_worldcover'] = (ee.ImageCollection('ESA/WorldCover/v100')
                                                     .first() # only one image is returned
                                                     .select('Map')
                                                     .remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                                                     .reproject(self.proj)
                                                     .unmask(no_data_values['ESA_WorldCover'])
                                                     .rename('ESA_WorldCover'))

        if self.pixel_level_data['esa_worldcover'].eq(no_data_values['ESA_WorldCover']).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=resolution).get('ESA_WorldCover').getInfo() == 1: # if all pixels in the tile are no data
            return False

    def precipitation_temperature(self):
        collection = 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
        bands = ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'total_precipitation_sum']
        year, month, _ = list(map(int, self.tile_level_data['sentinel2_date'].split('-')))
        month_first_day = datetime(year, month, 1)
        month_last_day = (month_first_day + relativedelta(months=1, days=-1)).strftime('%Y-%m-%d')
        last_month = month - 1 if month > 1 else 12
        last_month_first_day = datetime(year, last_month, 1)
        last_month_last_day = (last_month_first_day + relativedelta(months=1, days=-1)).strftime('%Y-%m-%d')
        precipitation_temperature_month = (ee.ImageCollection(collection)
                                             .filterDate(month_first_day, month_last_day)
                                             .select(bands)
                                             .toBands()
                                             .rename(['temperature_month_mean', 'temperature_month_min', 'temperature_month_max', 'precipitation_month']))
        precipitation_temperature_last_month = (ee.ImageCollection(collection)
                                                  .filterDate(last_month_first_day, last_month_last_day)
                                                  .select(bands)
                                                  .toBands()
                                                  .rename(['temperature_last_month_mean', 'temperature_last_month_min', 'temperature_last_month_max', 'precipitation_last_month']))
        last_year_next_day = (datetime.strptime(month_last_day, '%Y-%m-%d') + relativedelta(years=-1, days=1)).strftime('%Y-%m-%d')
        precipitation_temperature_year = ee.ImageCollection(collection).filterDate(last_year_next_day, month_last_day)
        band_reducer = {'temperature_2m': ee.Reducer.mean(),
                        'temperature_2m_min': ee.Reducer.min(),
                        'temperature_2m_max': ee.Reducer.max(),
                        'total_precipitation_sum': ee.Reducer.sum()}
        precipitation_temperature_year = ee.ImageCollection([precipitation_temperature_year.select(band).reduce(band_reducer[band]) for band in bands]).toBands().rename(['temperature_year_mean', 'temperature_year_min', 'temperature_year_max', 'precipitation_year']).float()
        self.precipitation_temperature = ee.Image.cat([precipitation_temperature_month, precipitation_temperature_last_month, precipitation_temperature_year]).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.tile, scale=resolution).getInfo()

    def precipitation(self):
        self.precipitation_temperature()
        precipitation = [no_data_values['precipitation'] if value is None else value for key, value in self.precipitation_temperature.items() if key.startswith('precipitation')] # last month, month, year
        self.tile_level_data['precipitation'] = json.dumps(precipitation)

        if all(value == no_data_values['precipitation'] for value in precipitation):
            return False

    def temperature(self):
        temperature = [no_data_values['temperature'] if value is None else value for key, value in self.precipitation_temperature.items() if key.startswith('temperature')] # last month max, mean, min; month max, mean, min; year max, mean, min
        self.tile_level_data['temperature'] = json.dumps(temperature)

        if all(value == no_data_values['temperature'] for value in temperature):
            return False

    def task_data(self):
        if self.task == 'biomass':
            self.pixel_level_data[self.task] = ee.Image.constant(-9999).paint(self.gedi_points, 'agbd').reproject(self.proj).rename(self.task) # image array with -9999 wherever there are no points, aligned with Sentinel-2
        elif 'soil' in self.task:
            self.tile_level_data[self.task] = self.point['properties'][self.task] # saves the soil value for the tile
        elif self.task == 'species':
            self.tile_level_data[self.task] = json.dumps(self.point['properties'][self.task]) # saves the species list for the tile

    def save_tiff(self, image):
        band_names = image.bandNames().getInfo()
        url = image.getDownloadUrl({'name': f'tile_{self.id}',
                                    'scale': resolution,
                                    'crs': self.crs,
                                    'region': self.tile,
                                    'format': 'GeoTIFF',
                                    'bands': band_names})
        success = False
        tiff_path = f'{data_dir_path}/{self.task}/tiffs/tile_{self.id}.tif'

        while not success:
            response = requests.get(url, stream=True, verify=certifi.where())

            if response.status_code != 200:
                continue # goes to the start of the loop

            with open(tiff_path, 'wb') as tiff: # writes in binary mode
                tiff.write(response.content)

            with rasterio.open(tiff_path) as tiff: # opens the TIFF
                array = tiff.read()

                with rasterio.open(tiff_path, 'w', **tiff.meta) as updated_tiff: # opens a new TIFF at the same location
                    updated_tiff.write(array)
                    updated_tiff.update_tags(**self.tile_level_data) # saves the tile-level data in the TIFF's tags

                    for i in range(tiff.count):
                        updated_tiff.update_tags(i+1, BAND_NAME=band_names[i]) # saves the band names

            success = True

if __name__ == '__main__':
    EEData(task=argv[1], point_id=int(argv[2]))
