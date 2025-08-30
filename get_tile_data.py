'''
get_tile_data.py
'''

# ============================================== IMPORTS ============================================== #

from sys import argv
import builtins
import ee
import functools
import json
import numpy as np
import os
import subprocess
import time
import utils

# ============================================== GLOBAL VARIABLES ============================================== #

ee.Initialize(project='mmearth-bench') # initializes EE with our project
data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']
env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
print = functools.partial(builtins.print, flush=True) # ensures print statements show up right away

# ============================================== FUNCTIONS ============================================== #

def get_tile_data(task):
    start_time = time.time()
    os.makedirs(f'{data_dir_path}/{task}/tiffs', exist_ok=True)
    points = utils.read_geojson(f'{data_dir_path}/{task}/{task}_points.geojson') # reading the GeoJSON file
    end_id = len(points['features'])
    tile_ids_run_path = f'{data_dir_path}/{task}/{task}_tile_ids_run.json'

    if os.path.exists(tile_ids_run_path): # if there is some data saved
        tile_ids_run = utils.read_json(tile_ids_run_path) # reads the existing missing modality data
    else:
        tile_ids_run = {i: 0 for i in range(end_id)} # initializes an empty dictionary

        # saves the dictionary as a JSON file
        with open(tile_ids_run_path, 'w') as file:
            json.dump(tile_ids_run, file, indent=4)

    tile_ids_to_run = [id_ for id_, status in tile_ids_run.items() if status == 0]

    for point_id in tile_ids_to_run:
        print(f'Processing tile {point_id}/{end_id-1}')

        while utils.count_running_jobs() > 29: # if more than 29 jobs are running
            time.sleep(1) # checks again after 1 second

        subprocess.run(['sbatch', '-t', '10', '-p', partitions, '--mem', '500M', '--job-name', f'{task}_get_tile_{point_id}_data', '-o', f'{data_dir_path}/{task}/output-files/get_tile_data/get_tile_{point_id}_data.out', '--account', 'gajos_lab', 'job.sh', env_path, 'ee_data.py', task, str(point_id)])

    print(f'Time taken: {utils.format_time(seconds=time.time()-start_time)}')

def check_complete(task):
    tile_count = len(utils.read_geojson(f'{data_dir_path}/{task}/{task}_points.geojson')['features'])
    tile_ids_run_status = np.array(list(utils.read_yaml(f'{data_dir_path}/{task}/{task}_tile_ids_run.json').values()))
    number_of_tiles_run = np.count_nonzero(tile_ids_run_status != 0)
    complete = tile_count == number_of_tiles_run

    print(f'All tiles run for {task}: {complete} ({number_of_tiles_run}/{tile_count})')

    tiles_not_missing = np.count_nonzero(tile_ids_run_status == 'done')
    print(f'{tiles_not_missing} tiles made')

    assert len(os.listdir(f'{data_dir_path}/{task}/tiffs')) == tiles_not_missing

# ============================================== RUN ============================================== #

if __name__ == '__main__':
    if 'check_complete' in argv[1]: # python get_tile_data.py check_complete TASK
        check_complete(argv[2])
    elif 'for' not in argv[1]: # python get_tile_data.py TASK
        partitions = utils.read_yaml('config-user.yml')['partitions'] # list of partition(s)
        env_path = utils.read_yaml('config-user.yml')['env_path'] # path to conda environment
        task = argv[1]
        subprocess.run(['sbatch', '-t', '1-00:00', '-p', partitions, '--mem', '500M', '--job-name', f'{task}_mmearth_modalities', '-o', f'{data_dir_path}/{task}/output-files/{task}_mmearth_modalities.out', 'job.sh', env_path, 'get_tile_data.py', f'for_{task}'])
    elif 'for' in argv[1]: # python get_tile_data.py for_TASK
        task = argv[1].split('for_')[1]
        print(f'Task = {task}')
        get_tile_data(task)
