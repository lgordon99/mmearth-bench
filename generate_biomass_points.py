'''
generate_biomass_points.py by Lucia Gordon
A script to create a GeoJSON with biomass points balanced across biomes
'''

# imports
from google.cloud import storage
from shapely import bounds
from sys import argv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ee
import geemap
import geopandas as gpd
import json
import math
import matplotlib.pyplot as plt
import os
import random
import subprocess
import sys
import time
import utils

random.seed(42)
ee.Initialize(project='mmearth-bench') # initializes EE with our project
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
client = storage.Client()
bucket = client.bucket('biomass_ecoregion_data')
OUTER_TILE_SIZE_M = utils.read_yaml('config.yml')['OUTER_TILE_SIZE_M']

def get_ecoregion_collection():
    return ee.FeatureCollection('RESOLVE/ECOREGIONS/2017').filter(ee.Filter.neq('BIOME_NAME', 'N/A')) # dataset with the extents of the 846 terrestrial ecoregions

def save_biome_ecoregion_labels():
    ecoregion_collection = get_ecoregion_collection()
    biomes = sorted(set(ecoregion_collection.aggregate_array('BIOME_NAME').getInfo())) # list of 14 biomes
    ecoregions = sorted(set(ecoregion_collection.aggregate_array('ECO_NAME').getInfo())) # list of 846 ecoregions

    # assign each biome and ecoregion an integer label
    for item in ['biome', 'ecoregion']:
        with open(f'{data_dir_path}/{item}_labels.json', 'w') as file:
            json.dump({name: i for i, name in enumerate(locals()[f'{item}s'])}, file, indent=4)

def get_ecoregions_in_gedi_collection():
    gedi_latitude_range = [-51.6, 51.6] # GEDI covers the latitude band between 51.6 degees N and S
    gedi_range_polygon = ee.Geometry.Polygon(coords=[[-180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[0]],
                                                    [180, gedi_latitude_range[1]],
                                                    [-180, gedi_latitude_range[1]]],
                                             proj=None,
                                             geodesic=False) # polygon covering GEDI range
    ecoregions_in_gedi_collection = (get_ecoregion_collection()
                                     .filterBounds(gedi_range_polygon) # excludes ecoregions outside of the GEDI range, leaving 774
                                     .map(lambda feature: feature.setGeometry(feature.geometry().intersection(gedi_range_polygon, maxError=1)))) # crops ecoregions to the GEDI range

    return ecoregions_in_gedi_collection

def get_biomass_tile_counts():
    '''
    The areas of the ecoregions are not perfectly reproducible because of the maxError in the intersection() function used to crop the ecoregions to the GEDI range
    The number of tiles for each ecoregion should be unaffected by this
    '''

    start_time = time.time()
    os.makedirs(f'{data_dir_path}/biomass', exist_ok=True)
    ecoregions_in_gedi_collection = get_ecoregions_in_gedi_collection() # collection of ecoregions cropped to the GEDI range
    biomes = sorted(set(ecoregions_in_gedi_collection.aggregate_array('BIOME_NAME').getInfo())) # list of 14 biomes
    biomes_ecoregions = {biome: {} for biome in biomes}
    num_tiles_per_biome = math.ceil(20000 / len(biomes)) # number of tiles per biome = 1,429
    intended_num_tiles = 0

    for i, biome in enumerate(biomes):
        print(f'Biome {i+1}/{len(biomes)}: {biome}')

        ecoregions = sorted(ecoregions_in_gedi_collection.filter(ee.Filter.eq('BIOME_NAME', biome)).aggregate_array('ECO_NAME').getInfo()) # list of ecoregions in biome
        biomes_ecoregions[biome]['ecoregions'] = {}

        print('Calculating areas')

        for j, ecoregion in enumerate(ecoregions):
            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')

            biomes_ecoregions[biome]['ecoregions'][ecoregion] = {'area': ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion)).geometry().area().getInfo()} # saves the ecoregion's name and area

        biomes_ecoregions[biome]['area'] = sum([biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] for ecoregion in ecoregions]) # sets the biome area to the sum of its ecoregions' areas

        print('Calculating tile numbers')

        for j, ecoregion in enumerate(ecoregions):
            print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')

            biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles'] = math.ceil(num_tiles_per_biome * biomes_ecoregions[biome]['ecoregions'][ecoregion]['area'] / biomes_ecoregions[biome]['area']) # number of tiles per ecoregion is proportional to the size of the ecoregion
            intended_num_tiles += biomes_ecoregions[biome]['ecoregions'][ecoregion]['num_tiles'] # adds the number of tiles for the ecoregion to the total number of tiles

    with open(f'{data_dir_path}/biomass/ecoregion_tile_counts.json', 'w') as file:
        json.dump(biomes_ecoregions, file, indent=4)

    print(f'Total number of tiles = {intended_num_tiles}') # 20,421 tiles
    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}') # ~13 minutes

def get_specific_ecoregion_collection(ecoregion):
    return get_ecoregions_in_gedi_collection().filter(ee.Filter.eq('ECO_NAME', ecoregion)) # extracts the collection for the ecoregion

def generate_ecoregion_points(biome, ecoregion, num_ecoregion_tiles):
    start_time = time.time()
    print(f'{num_ecoregion_tiles} tile(s) in ecoregion')
    # os.makedirs(f'{data_dir_path}/biomass/points/{biome.replace("/", "_")}', exist_ok=True) # TEMPORARILY HERE

    biome_labels = utils.read_json(f'{data_dir_path}/biome_labels.json') # reads the biome labels
    ecoregion_labels = utils.read_json(f'{data_dir_path}/ecoregion_labels.json') # reads the ecoregion labels
    ecoregion_collection = get_specific_ecoregion_collection(ecoregion) # extracts the collection for the ecoregion
    gedi_points_in_ecoregion = (utils.get_gedi_points(ecoregion_collection).map(lambda point: ee.Feature(point.geometry())) # only keeps geometry information
                                                                           .randomColumn(seed=42) # adds a new property to each feature containing a random number
                                                                           .sort('random') # sorts the features by the random numbers
                                                                           .limit(2*num_ecoregion_tiles) # selects twice as many points as needed, possibly fewer than the desired amount depending on how many points are available
                                                                           .map(lambda point: point.set('outer_tile', point.buffer(OUTER_TILE_SIZE_M / 2).bounds().geometry()))) # creates outer tiles around the points
    geojson_prefix = f'{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}_biomass_points'
    task = ee.batch.Export.table.toCloudStorage(collection=gedi_points_in_ecoregion,
                                                bucket='biomass_ecoregion_data',
                                                fileNamePrefix=geojson_prefix,
                                                fileFormat='GeoJSON')
    task.start()
    blob = bucket.blob(f'{geojson_prefix}.geojson')

    while not blob.exists():
        time.sleep(10) # waits 10 seconds

    geojson_path = f'{data_dir_path}/biomass/points/{geojson_prefix}.geojson'
    blob.download_to_filename(geojson_path)
    points = utils.read_geojson(geojson_path)['features']

    if len(points) < num_ecoregion_tiles:
        print(f'The ecoregion had only {len(points)} valid GEDI points')
        print(f'This is less than the desired number of tiles')

    if len(points) < 2 * num_ecoregion_tiles:
        print(f'The ecoregion had only {len(points)} valid GEDI points')
        print(f'This is less than twice the desired number of tiles')

    print(f'{len(points)} points before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    print(f'{len(points)} points after removing overlaps')

    random.shuffle(points) # shuffles the points list
    points = points[:num_ecoregion_tiles] # selects the first num_ecoregion_tiles points

    if len(points) < num_ecoregion_tiles:
        print(f'Fewer points were made than desired')

    points = [{**{key: value for key, value in point.items() if key != 'id'}, 'properties': {**{key: value for key, value in point['properties'].items() if key != 'random'}, 'biome': biome_labels[biome], 'ecoregion': ecoregion_labels[ecoregion]}} for point in points] # removes the 'id' and 'random' properties from the points and adds biome and ecoregion labels
    utils.save_geojson(points, path=geojson_path)

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def generate_biomass_points():
    start_time = time.time()
    ecoregion_tile_counts = utils.read_json(f'{data_dir_path}/biomass/ecoregion_tile_counts.json')
    biomes = ecoregion_tile_counts.keys()
    complete = False

    while not complete:
        complete = True

        for i, biome in enumerate(biomes):
            os.makedirs(f'{data_dir_path}/biomass/points/{biome.replace("/", "_")}', exist_ok=True)
            ecoregions = ecoregion_tile_counts[biome]['ecoregions'].keys()

            for j, ecoregion in enumerate(ecoregions):
                points_path = f'{data_dir_path}/biomass/points/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}_biomass_points.geojson'
                run_ecoregion = True

                if os.path.exists(points_path): # if there is a points file saved
                    run_ecoregion = False # ecoregion has already been run
                else: # if there is no points file saved
                    complete = False # at least one ecoregion is missing data
                    output_file_path = f'{data_dir_path}/biomass/output-files/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.out'
                    error_file_path = f'{data_dir_path}/biomass/error-files/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err'

                    if os.path.exists(error_file_path): # if there is an error file
                        with open(error_file_path, 'r') as file: # opens error file
                            error = file.read()

                            if len(error) > 0: # if there is an error
                                print('\nError = {}'.format(error.split("\n")[-2]))
                            else: # if there is no error
                                run_ecoregion = False # ecoregion is currently being run

                if run_ecoregion:
                    while utils.count_running_jobs() > 40: # if more than 40 jobs are running
                        time.sleep(1) # checks again after 1 second

                    print(f'Biome {i+1}/{len(biomes)}: {biome}')
                    print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion}')
                    sys.stdout.flush() # ensures that the print statements are displayed immediately
                    num_ecoregion_tiles = ecoregion_tile_counts[biome]['ecoregions'][ecoregion]['num_tiles']
                    subprocess.run(['sbatch', '-t', f'03:00:00', '-p', partitions, '--mem', '500M', '--job-name', f'biome_{biome.replace("/", "_")}_ecoregion_{ecoregion.replace("/", "_")}', '-o', output_file_path, '-e', error_file_path, 'job.sh', env_path, 'generate_biomass_points.py', 'generate_ecoregion_points', biome.replace(' ', '_'), ecoregion.replace(' ', '_'), str(num_ecoregion_tiles)])
                    time.sleep(50) # checks again after 50 seconds since there is a delay between the job being submitted and the job running

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def merge_ecoregion_points():
    points = []

    for biome in os.listdir(f'{data_dir_path}/biomass/points'):
        for ecoregion_points_file in os.listdir(f'{data_dir_path}/biomass/points/{biome}'):
            points += utils.read_geojson(f'{data_dir_path}/biomass/points/{biome}/{ecoregion_points_file}')['features']

    print(f'{len(points)} points before removing overlaps')

    points = utils.remove_overlapping_tiles(points) # removes points with overlapping tiles
    random.shuffle(points) # shuffles the points list
    points = [{**point, 'id': i} for i, point in enumerate(points)] # assigns each point an ID
    print(f'{len(points)} points after removing overlaps')

    utils.save_geojson(features=points, path=f'{data_dir_path}/biomass/biomass_points.geojson') # saves the points

def check_biomass_points():
    ecoregion_tile_counts = utils.read_json(f'{data_dir_path}/biomass/ecoregion_tile_counts.json')
    biomes = ecoregion_tile_counts.keys()
    # num_missing_ecoregions = 0
    # num_ecoregions_missing_tiles = 0
    num_ecoregions = 0
    missing_ecoregions = []
    ecoregions_missing_tiles = []

    for i, biome in enumerate(biomes):
        ecoregions = ecoregion_tile_counts[biome]['ecoregions'].keys()
        print(f'\nBiome {i+1}/{len(biomes)}: {biome}')

        for j, ecoregion in enumerate(ecoregions):
            points_path = f'{data_dir_path}/biomass/points/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}_biomass_points.geojson'
            error_file_path = f'{data_dir_path}/biomass/error-files/{biome.replace("/", "_")}/{ecoregion.replace("/", "_")}.err'
            num_ecoregion_tiles = ecoregion_tile_counts[biome]['ecoregions'][ecoregion]['num_tiles']
            num_ecoregions += 1

            if not os.path.exists(points_path):
                # num_missing_ecoregions += 1
                missing_ecoregions.append(ecoregion)

                print(f'\nEcoregion {j+1}/{len(ecoregions)}: {ecoregion} tiles do not exist ({num_ecoregion_tiles} tiles)')

                with open(error_file_path, 'r') as file:
                    content = file.read()

                    if len(content) > 0:
                        print(content.split('\n')[-2])
            elif len(utils.read_geojson(points_path)['features']) < num_ecoregion_tiles:
                # num_ecoregions_missing_tiles += 1
                ecoregions_missing_tiles.append(ecoregion)

                # _, gedi_points = get_gedi_points(ecoregion)

                # with open(error_file_path, 'r') as file:
                #     content = file.read()

                    # if len(content) > 0:
                    #     print(content.split('\n')[-2])

                # print(f'Ecoregion {j+1}/{len(ecoregions)}: {ecoregion} is missing tiles ({len(utils.read_geojson(path)["features"])}/{num_ecoregion_tiles} tile(s) made, {gedi_points.size().getInfo()} GEDI points)')
                print(f'\nEcoregion {j+1}/{len(ecoregions)}: {ecoregion} is missing tiles ({len(utils.read_geojson(path)["features"])}')

    print(f'\n{len(missing_ecoregions)}/{num_ecoregions} ecoregions are missing')
    print(f'{len(ecoregions_missing_tiles)}/{num_ecoregions} ecoregions are missing tiles')

    # ecoregions_in_gedi_collection = get_ecoregions_in_gedi_collection()
    # fig = plt.figure(dpi=300)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_title('Ecoregions with Missing Tiles', fontsize=8)
    # ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.3)
    # ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=0.3)

    # # for ecoregion in missing_ecoregions:
    # #     ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
    # #     ecoregion_gdf = gpd.GeoDataFrame.from_features(geemap.ee_to_geojson(ecoregion_collection))
    # #     ecoregion_gdf.plot(ax=ax, edgecolor='blue', facecolor='blue', transform=ccrs.PlateCarree())

    # for ecoregion in ecoregions_missing_tiles:
    #     ecoregion_collection = ecoregions_in_gedi_collection.filter(ee.Filter.eq('ECO_NAME', ecoregion))
    #     ecoregion_gdf = gpd.GeoDataFrame.from_features(geemap.ee_to_geojson(ecoregion_collection))
    #     ecoregion_gdf.plot(ax=ax, edgecolor='red', facecolor='red', transform=ccrs.PlateCarree())

    # ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    # # legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Missing ecoregions'),
    # #                    Line2D([0], [0], color='red', lw=2, label='Ecoregions missing tiles')]

    # # ax.legend(handles=legend_elements, fontsize=6)
    # plt.savefig('biomass/figures/ecoregions_missing_tiles.png', bbox_inches='tight')

if __name__ == '__main__':
    if not os.path.exists(f'{data_dir_path}/biome_labels.json'):
        save_biome_ecoregion_labels()

    if not os.path.exists(f'{data_dir_path}/biomass/ecoregion_tile_counts.json'):
        get_biomass_tile_counts()

    if not bucket.exists():
        bucket = client.create_bucket(bucket)

    if len(argv) == 1:
        subprocess.run(['sbatch', '-t', '7-00:00:00', '-p', partitions, '--mem', '500M', '--job-name', 'generate_biomass_points', '-o', f'{data_dir_path}/biomass/output-files/generate_biomass_points.out', '-e', f'{data_dir_path}/biomass/error-files/generate_biomass_points.err', 'job.sh', env_path, 'generate_biomass_points.py', 'generate_biomass_points'])
    if len(argv) > 1:
        if argv[1] == 'generate_biomass_points':
            generate_biomass_points()
        elif argv[1] == 'generate_ecoregion_points':
            generate_ecoregion_points(biome=argv[2].replace('_', ' '), ecoregion=argv[3].replace('_', ' '), num_ecoregion_tiles=int(argv[4]))
        elif argv[1] == 'merge_ecoregion_points':
            merge_ecoregion_points()
        elif argv[1] == 'check_biomass_points':
            check_biomass_points()

    # ecoregion_tile_counts = utils.read_json(f'{data_dir_path}/biomass/ecoregion_tile_counts.json')
    # biomes = ecoregion_tile_counts.keys()

    # for i, biome in enumerate(biomes):
    #     ecoregions = ecoregion_tile_counts[biome]['ecoregions'].keys()

    #     for j, ecoregion in enumerate(ecoregions):
    #         num_ecoregion_tiles = ecoregion_tile_counts[biome]['ecoregions'][ecoregion]['num_tiles']

    #         # if ecoregion in ['Yunnan Plateau subtropical evergreen forests', 'Araucaria moist forests', 'Northeast Congolian lowland forests', 'Upper Gangetic Plains moist deciduous forests', 'Xingu-Tocantins-Araguaia moist forests', 'Central Congolian lowland forests', 'Peruvian Yungas', 'South China-Vietnam subtropical evergreen forests', 'Madeira-Tapajós moist forests', 'Northern Indochina subtropical forests', 'Jian Nan subtropical evergreen forests', 'Tapajós-Xingu moist forests', 'Western Guinean lowland forests', 'Bahia interior forests', 'Alto Paraná Atlantic forests', 'Northwest Congolian lowland forests', 'Petén-Veracruz moist forests', 'Southern Swahili coastal forests and woodlands', 'Southwest Amazon moist forests', 'Mato Grosso tropical dry forests', 'Zambezian mopane woodlands', 'Kimberly tropical savanna', 'Angolan wet miombo woodlands', 'Northern Congolian Forest-Savanna', 'West Sudanian savanna', 'Guinean forest-savanna', 'Einasleigh upland savanna', 'Carpentaria tropical savanna', 'Angolan mopane woodlands', 'Southern Congolian forest-savanna', 'Sahelian Acacia savanna', 'East Sudanian savanna', 'Cerrado', 'Uruguayan savanna', 'Somali Acacia-Commiphora bushlands and thickets', 'Brigalow tropical savanna', 'Central Zambezian wet miombo woodlands', 'Dry miombo woodlands', 'Victoria Plains tropical savanna', 'Zambezian-Limpopo mixed woodlands', 'Zambezian Baikiaea woodlands', 'Dry Chaco', 'Horn of Africa xeric bushlands', 'Mitchell Grass Downs', 'Central bushveld', 'Zambezian flooded grasslands', 'Piney Woods', 'British Columbia coastal conifer forests', 'Alps conifer and mixed forests', 'Arizona Mountains forests', 'Colorado Rockies forests', 'Northern Rockies conifer forests', 'South Central Rockies forests', 'Altai montane forest and forest steppe', 'Sayan montane conifer forests', 'Central Pacific Northwest coastal forests', 'Da Hinggan-Dzhagdy Mountains conifer forests', 'Carpathian montane forests', 'Okanogan dry forests', 'Central-Southern Cascades Forests', 'Mediterranean woodlands and forests', 'Italian sclerophyllous and semi-deciduous forests', 'Southwest Australia savanna', 'Iberian sclerophyllous and semi-deciduous forests', 'Tyrrhenian-Adriatic sclerophyllous and mixed forests', 'Aegean and Western Turkey sclerophyllous and mixed forests', 'California interior chaparral and woodlands', 'Murray-Darling woodlands and mallee', 'Mediterranean dry woodlands and steppe', 'Coolgardie woodlands', 'Eastern Mediterranean conifer-broadleaf forests', 'Central Indochina dry forests', 'Khathiar-Gir dry deciduous forests', 'Chiquitano dry forests', 'Caatinga', 'Appalachian mixed mesophytic forests', 'Caucasus mixed forests', 'Central European mixed forests', 'Upper Midwest US forest-savanna transition', 'Valdivian temperate forests', 'Taiheiyo evergreen forests', 'Appalachian-Blue Ridge forests', 'Central Anatolian steppe and woodlands', 'Western European broadleaf forests', 'New England-Acadian forests', 'Eastern Anatolian deciduous forests', 'Manchurian mixed forests', 'Eastern Canadian Forest-Boreal transition', 'European Atlantic mixed forests', 'Balkan mixed forests', 'Western Great Lakes forests', 'Pannonian mixed forests', 'Eastern Great Lakes lowland forests', 'Northeast China Plain deciduous forests', 'Zagros Mountains forest steppe', 'Southeast Australia temperate forests', 'East European forest steppe', 'Central China Loess Plateau mixed forests', 'Changjiang Plain evergreen forests', 'Eastern Australian temperate forests', 'Huang He Plain mixed forests', 'Ussuri broadleaf and mixed forests', 'Southern Great Lakes forests', 'Western shortgrass prairie', 'Mongolian-Manchurian grassland', 'Southeast US conifer savannas', 'Syrian xeric grasslands and shrublands', 'Altai steppe and semi-desert', 'Northern Tallgrass prairie', 'Selenge-Orkhon forest steppe', 'Daurian forest steppe', 'Alai-Western Tian Shan steppe', 'Kazakh forest steppe', 'Eastern Anatolian montane steppe', 'Southeast Australia temperate savanna', 'Canadian Aspen forests and parklands', 'Palouse prairie', 'Humid Pampas', 'Kazakh upland steppe', 'Kazakh steppe', 'Patagonian steppe', 'Northern Shortgrass prairie', 'Gissaro-Alai open woodlands', 'Eastern Australia mulga shrublands', 'Central Tallgrass prairie', 'Central-Southern US mixed grasslands', 'Low Monte', 'Montana Valley and Foothill grasslands', 'Tian Shan foothill arid steppe', 'Central US forest-grasslands transition', 'Pontic steppe', 'Espinal', 'Sierra Madre Occidental pine-oak forests', 'Central Andean puna', 'Khangai Mountains alpine meadow', 'Tibetan Plateau alpine shrublands and meadows', 'Ordos Plateau steppe', 'High Monte', 'Highveld grasslands', 'Altai alpine meadow and tundra', 'Tian Shan montane steppe and meadows', 'Southeast Tibet shrublands and meadows', 'Sayan alpine meadows and tundra', 'North Tibetan Plateau-Kunlun Mountains alpine desert', 'Central Andean dry puna', 'Central Tibetan Plateau alpine steppe', 'Great Lakes Basin desert steppe', 'Nama Karoo shrublands', 'Meseta Central matorral', 'Sonoran desert', 'Thar desert', 'Qaidam Basin semi-desert', 'Baluchistan xeric woodlands', 'Great Victoria desert', 'Western Australian Mulga shrublands', 'Eastern Gobi desert steppe', 'Colorado Plateau shrublands', 'North Arabian desert', 'South Iran Nubo-Sindian desert and semi-desert', 'Gobi Lakes Valley desert steppe', 'South Sahara desert', 'Wyoming Basin shrub steppe', 'Alashan Plateau semi-desert', 'Simpson desert', 'Kazakh semi-desert', 'Caspian lowland desert', 'Chihuahuan desert', 'Tirari-Sturt stony desert', 'Snake-Columbia shrub steppe', 'Junggar Basin semi-desert', 'Central Asian northern desert', 'Aravalli west thorn scrub forests', 'North Saharan Xeric Steppe and Woodland', 'Gibson desert', 'Kalahari xeric savanna', 'Arabian desert', 'Great Sandy-Tanami desert', 'Badghyz and Karabil semi-desert', 'Central Afghan Mountains xeric woodlands', 'Taklimakan desert', 'Great Basin shrub steppe', 'Central Asian riparian woodlands', 'Central Ranges xeric scrub', 'Nullarbor Plains xeric shrublands', 'Central Persian desert basins', 'Central Asian southern desert', 'Trans-Baikal conifer forests', 'Central Canadian Shield forests', 'Southern Hudson Bay taiga', 'Midwest Canadian Shield forests', 'Eastern Canadian forests', 'Okhotsk-Manchurian taiga']:
    #         if ecoregion == 'Northeast Congolian lowland forests':
    #             print(ecoregion)
    #             generate_ecoregion_points(biome, ecoregion, num_ecoregion_tiles)
