# ============================================== IMPORTS ============================================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import LinearSegmentedColormap
from sys import argv
from tqdm import tqdm

# ============================================== GLOBAL VARIABLES ============================================== #

num_decimals = 1
models = ['ResNet50', 'ResNet50-ImageNet', 'DINOv2', 'MPMAE', 'MPMAE-MMEarth']
splits = ['Random', 'Geographic']
adaptation_methods = ['lp', 'ft', 'llrd']

# ============================================== FUNCTIONS ============================================== #

def get_run_data(run, hyperparameters):
    if run.state != 'finished':
        print(f'{run.name} is not finished yet')
        exit()

    run_data = {'name': run.name,
                'Val RMSE': round(run.history(keys=['Val RMSE'])['Val RMSE'].min(), num_decimals),
                'Random test RMSE': round(run.summary_metrics['Random test RMSE/dataloader_idx_0'], num_decimals),
                'Random test R2': round(run.summary_metrics['Random test R2/dataloader_idx_0'], num_decimals),
                'Random test MAE': round(run.summary_metrics['Random test MAE/dataloader_idx_0'], num_decimals),
                'Random test ME': round(run.summary_metrics['Random test ME/dataloader_idx_0'], num_decimals),
                'Geographic test RMSE': round(run.summary_metrics['Geographic test RMSE/dataloader_idx_1'], num_decimals),
                'Geographic test R2': round(run.summary_metrics['Geographic test R2/dataloader_idx_1'], num_decimals),
                'Geographic test MAE': round(run.summary_metrics['Geographic test MAE/dataloader_idx_1'], num_decimals),
                'Geographic test ME': round(run.summary_metrics['Geographic test ME/dataloader_idx_1'], num_decimals)}
    run_hyperparameter_data = {hyperparameter: run.config[hyperparameter] for hyperparameter in hyperparameters}
    run_data = {**run_data, **run_hyperparameter_data}

    return run_data

def monitor_sweep(name):
    sweep_log_df = pd.read_csv('sweep_log.csv')
    sweep_ID = sweep_log_df[sweep_log_df['name'] == name]['sweep_ID'].values[0]
    sweep = wandb.Api().sweep(sweep_ID)
    hyperparameters = sweep.config['parameters'].keys()
    all_run_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_run_data, run, hyperparameters) for run in sweep.runs]
        all_run_data = [future.result() for future in futures if future.result() is not None]

    df = pd.DataFrame(all_run_data).sort_values(by='Val RMSE')
    best_run = df.iloc[0]

    return best_run['Random test RMSE'], best_run['Geographic test RMSE']
    # return best_run

def create_table(task):
    results = np.full((len(models), len(splits) * len(adaptation_methods)), np.nan)

    for i, model in enumerate(tqdm(models, desc='Models', position=0)):
        for j, adaptation_method in enumerate(tqdm(adaptation_methods, desc='Adaptation methods', position=1, leave=False)):
            model_name = model.lower().replace('-', '_')
            random_test_rmse, geographic_test_rmse = monitor_sweep(name=f'{task}_{model_name}_{adaptation_method}')
            results[i, j] = random_test_rmse
            results[i, j + len(adaptation_methods)] = geographic_test_rmse

    columns = pd.MultiIndex.from_tuples([('Random', 'LP'), ('Random', 'FT'),
                                         ('Random', 'LLRD'), ('Geographic', 'LP'),
                                         ('Geographic', 'FT'), ('Geographic', 'LLRD')])
    df = pd.DataFrame(results, index=models, columns=columns)

    print(df)

create_table(task=argv[1])
